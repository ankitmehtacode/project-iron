"""Score V-JEPA2 semantics against v3-indoor's GT tracks, gate first.

The Day-10 appearance gate already refuses v3 on measured evidence (texture
energy 3.03 against a floor of 12.0). Day 11 tests that prediction: if the
partition (geometry-derived vs appearance-learned) generalises to semantics,
these metrics must degenerate on v3.

Three metrics, all along GT agent tracks (no CoTracker3 dependency — the
tracks are known):

- **Same-object retrieval mAP.** For every (track, frame) embedding, retrieve
  its nearest neighbours across every (track, frame) in the pool. Label:
  same track. If the encoder distinguishes tracks at all, mAP is well above
  chance; on degenerate input it collapses to chance (1 / n_tracks).
- **Temporal embedding stability.** Mean cosine similarity between the
  embedding at frame f and f+1 along the same GT track. Low similarity
  means the encoder is jittering, high means it is either seeing a real
  object it recognises OR it is producing a near-constant output on
  featureless input — the two are distinguished by comparing across tracks.
- **Patch-boundary discontinuity.** For each pair of adjacent patches on the
  token grid, the L2 distance between their embeddings, averaged. A large
  value indicates checkerboarding (a common failure mode of over-aggressive
  positional embeddings on flat input).

Every metric is computed twice: once with the sidecar preprocessing spec
applied (the production path — mean/std standardisation from
``models/export/…/vjepa2_vitl_fp32.preprocess.json``), and once with the raw
[0, 1] clip fed straight in (reconstructing the pre-fix path behind a flag
without un-fixing production). The delta between the two is reported even
when the absolute numbers are degenerate: a relative improvement on
degenerate data is weak evidence, and labeling it that way is better than
dropping it.

    python scripts/eval_semantics.py
    python scripts/eval_semantics.py --limit 4

Exit codes:
    0  gates ran and metrics were computed on each path.
    1  weights, IR, spec, or clips missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.contracts import FrameGeometry  # noqa: E402
from src.contracts.tokens import TemporalSpan  # noqa: E402
from src.data import validity  # noqa: E402
from src.data.golden import load_golden_set  # noqa: E402
from src.models.preprocess import PreprocessSpec  # noqa: E402
from src.semantics.patch_mapping import (  # noqa: E402
    map_tracks_to_embeddings,
    tokens_from_encoder_output,
)

DEFAULT_IR = Path("models/export/2026-07-31/vjepa2_vitl_fp32.xml")


def resize_and_scale_clip(rgb_uint8: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Resize a clip to the spec's resolution and scale to [0, 1] float32.

    Deliberately does NOT standardise here — standardisation is what we are
    ablating. Returned shape ``[1, T, 3, H, W]``. Uses OpenCV so aspect and
    interpolation are the same as the wrapper.
    """
    import cv2

    height, width = spec.resolution
    resized = np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in rgb_uint8]
    )  # [T, H, W, 3]
    scaled = resized.astype(np.float32) / 255.0
    # [T, H, W, 3] -> [1, T, 3, H, W]
    return scaled.transpose(0, 3, 1, 2)[None]


def encode(
    infer: Any,
    clip_bt_chw: np.ndarray,
    spec: PreprocessSpec,
    standardise: bool,
) -> np.ndarray:
    """Run V-JEPA2 IR on one clip, honouring or bypassing the standardiser.

    ``standardise=False`` reconstructs the pre-fix path — raw [0, 1] into the
    encoder — without touching the wrapper the pipeline uses in production.
    """
    if standardise:
        prepped = spec.apply(clip_bt_chw)
    else:
        prepped = clip_bt_chw.astype(np.float32)
    infer.infer({"video": prepped})
    return infer.get_output_tensor(0).data.copy()


def embed_gt_tracks_for_clip(
    infer: Any,
    spec: PreprocessSpec,
    rgb: np.ndarray,
    gt_uv: np.ndarray,
    gt_occluded: np.ndarray,
    *,
    standardise: bool,
) -> np.ndarray:
    """Embedding per (track, frame). Shape: ``[N_tracks, T_frames, dim]``.

    Runs the encoder on the first ``spec.frames`` of the clip and looks up
    each GT track UV in the patch grid. GT UVs are in the ORIGINAL raster;
    scaled here to the encoder input raster so the patch grid geometry is
    the geometry the encoder actually sees.
    """
    orig_h, orig_w = rgb.shape[1], rgb.shape[2]
    encoder_h, encoder_w = spec.resolution
    sx, sy = encoder_w / orig_w, encoder_h / orig_h

    clip_frames = min(spec.frames, rgb.shape[0])
    clip = resize_and_scale_clip(rgb[:clip_frames], spec)
    features = encode(infer, clip, spec, standardise=standardise)

    grid = (encoder_h // spec.patch_size, encoder_w // spec.patch_size)
    span = TemporalSpan(
        start_ts_ns=0,
        end_ts_ns=max(1, clip_frames),
        frames_covered=clip_frames,
        tubelet=spec.tubelet,
    )
    tokens = tokens_from_encoder_output(
        features,
        batch_index=0,
        span=span,
        grid=grid,
        geometry=FrameGeometry(width=encoder_w, height=encoder_h),
        encoder_sha="ablation",
    )

    tracks = gt_uv[:clip_frames].copy().astype(np.float32)
    tracks[..., 0] *= sx
    tracks[..., 1] *= sy
    # [T, N, 2] as map_tracks_to_embeddings expects
    embeddings = map_tracks_to_embeddings(tokens, tracks)  # [T, N, dim] L2-normalised
    # Return [N, T, dim] for retrieval indexing convenience.
    return np.transpose(embeddings, (1, 0, 2))


def compute_metrics(
    per_clip_embeddings: list[dict[str, Any]],
    grid_size: tuple[int, int],
    encoder_output_grid: np.ndarray | None,
) -> dict[str, float]:
    """Same-object mAP, temporal stability, patch-boundary discontinuity.

    ``per_clip_embeddings`` is a list of ``{"clip_id", "track_id", "emb": [T, dim]}``
    — one entry per GT track across the whole scored set. That structure
    makes track identity globally unique so mAP is not gamed by identical
    intra-clip labels.
    """
    # Flatten to a global pool of embeddings and a label vector for retrieval.
    pool = []
    labels = []
    for entry in per_clip_embeddings:
        emb = entry["emb"]  # [T, dim]
        for frame in range(emb.shape[0]):
            pool.append(emb[frame])
            labels.append(entry["global_track_id"])
    pool_arr = np.stack(pool)  # [Q, dim]
    labels_arr = np.array(labels)  # [Q]
    n_queries = pool_arr.shape[0]

    # mAP over the pool. Skip degenerate case where every track has exactly
    # one entry (self-match only).
    if n_queries < 2 or len(set(labels)) < 2:
        map_score = float("nan")
    else:
        # Cosine sim: embeddings are L2-normalised by map_tracks_to_embeddings.
        sim = pool_arr @ pool_arr.T
        np.fill_diagonal(sim, -np.inf)  # exclude self
        order = np.argsort(-sim, axis=1)
        ranked_labels = labels_arr[order]
        matches = ranked_labels == labels_arr[:, None]
        # Average precision per query.
        aps = []
        for q in range(n_queries):
            hit_mask = matches[q]
            n_pos = int(hit_mask.sum())
            if n_pos == 0:
                continue
            precisions = np.cumsum(hit_mask, dtype=np.float64) / (np.arange(len(hit_mask)) + 1)
            ap = float((precisions * hit_mask).sum() / n_pos)
            aps.append(ap)
        map_score = float(np.mean(aps)) if aps else float("nan")

    # Temporal stability: cos(emb[f], emb[f+1]) along each track.
    stab = []
    for entry in per_clip_embeddings:
        emb = entry["emb"]
        if emb.shape[0] < 2:
            continue
        cos = np.sum(emb[:-1] * emb[1:], axis=1)  # already L2-normalised
        stab.append(float(cos.mean()))
    temporal_stability = float(np.mean(stab)) if stab else float("nan")

    # Patch-boundary discontinuity: on a single representative clip's tokens.
    if encoder_output_grid is not None:
        grid = encoder_output_grid  # [n_temporal, n_spatial, dim] — first clip
        n_temporal, n_spatial, dim = grid.shape
        rows, cols = grid_size
        assert rows * cols == n_spatial, (rows, cols, n_spatial)
        grid_rc = grid.reshape(n_temporal, rows, cols, dim)
        # L2-normalise for a comparable distance across paths.
        norm = np.linalg.norm(grid_rc, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        grid_norm = grid_rc / norm
        horiz = np.linalg.norm(grid_norm[:, :, 1:] - grid_norm[:, :, :-1], axis=-1)
        vert = np.linalg.norm(grid_norm[:, 1:] - grid_norm[:, :-1], axis=-1)
        boundary = float((horiz.mean() + vert.mean()) / 2.0)
    else:
        boundary = float("nan")

    n_labels = len(set(labels))
    chance_map = 1.0 / n_labels if n_labels > 0 else float("nan")
    return {
        "n_queries": n_queries,
        "n_tracks": n_labels,
        "mAP": map_score,
        "chance_mAP": chance_map,
        "temporal_cosine": temporal_stability,
        "patch_boundary_l2": boundary,
    }


def patch_visit_stats(clips_data: list[dict[str, Any]], spec: PreprocessSpec) -> dict[str, Any]:
    """How many unique patches each GT track occupies over the encoder window.

    If tracks stay inside one patch for the whole window, "same-object
    retrieval mAP" collapses to "look up the same embedding a few times"
    and the number stops describing anything about the encoder. This is
    the [[check-the-measuring-apparatus]] check for the semantic metrics:
    the score is confounded if the fixture cannot make tracks TRAVEL
    across patches.
    """
    encoder_h, encoder_w = spec.resolution
    frames = spec.frames
    patch = spec.patch_size
    per_track_unique = []
    for clip in clips_data:
        rgb = clip["rgb"]
        gt_uv = clip["track_uv"]
        sx, sy = encoder_w / rgb.shape[2], encoder_h / rgb.shape[1]
        scaled = gt_uv[: min(frames, gt_uv.shape[0])] * np.array([sx, sy])
        patches = np.floor(scaled / patch).astype(int)  # [T, N, 2]
        for n in range(patches.shape[1]):
            uniq = len({tuple(p) for p in patches[:, n]})
            per_track_unique.append(uniq)
    arr = np.asarray(per_track_unique)
    return {
        "n_tracks": int(arr.size),
        "unique_patches_mean": float(arr.mean()) if arr.size else float("nan"),
        "unique_patches_max": int(arr.max()) if arr.size else 0,
        "fraction_in_one_patch": float((arr == 1).mean()) if arr.size else float("nan"),
        "fraction_in_at_most_two_patches": float((arr <= 2).mean()) if arr.size else float("nan"),
    }


def run_path(
    infer: Any,
    spec: PreprocessSpec,
    clips_data: list[dict[str, Any]],
    *,
    standardise: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute all three metrics for one standardise-ON/OFF path."""
    per_clip = []
    grid_for_boundary = None
    global_id = 0
    grid_size = (spec.resolution[0] // spec.patch_size, spec.resolution[1] // spec.patch_size)
    for clip in clips_data:
        rgb = clip["rgb"]
        gt_uv = clip["track_uv"]
        gt_occ = clip["track_occluded"]
        n_tracks = gt_uv.shape[1]
        # [N, T, dim]
        embeddings = embed_gt_tracks_for_clip(
            infer, spec, rgb, gt_uv, gt_occ, standardise=standardise
        )
        # Restrict to frames where the GT track head is in-frame and not
        # occluded — a semantic embedding at an off-sensor pixel is a
        # look-up into the boundary patch, which has nothing to do with the
        # track's true content.
        clip_frames_used = embeddings.shape[1]
        for n in range(n_tracks):
            in_frame = (
                (gt_uv[:clip_frames_used, n, 0] >= 0)
                & (gt_uv[:clip_frames_used, n, 0] < rgb.shape[2])
                & (gt_uv[:clip_frames_used, n, 1] >= 0)
                & (gt_uv[:clip_frames_used, n, 1] < rgb.shape[1])
            )
            keep = in_frame & (~gt_occ[:clip_frames_used, n])
            if keep.sum() < 2:
                continue
            per_clip.append(
                {
                    "clip_id": clip["clip_id"],
                    "global_track_id": global_id,
                    "emb": embeddings[n, keep, :],
                }
            )
            global_id += 1
        if grid_for_boundary is None:
            # Recompute for the first clip only, to isolate one grid.
            clip_bt_chw = resize_and_scale_clip(rgb[: spec.frames], spec)
            features = encode(infer, clip_bt_chw, spec, standardise=standardise)
            n_spatial = grid_size[0] * grid_size[1]
            total, dim = features[0].shape
            grid_for_boundary = features[0].reshape(total // n_spatial, n_spatial, dim)

    metrics = compute_metrics(per_clip, grid_size, grid_for_boundary)
    return metrics, per_clip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--set", dest="set_version", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/day11/semantics.json"))
    args = parser.parse_args(argv)

    if not args.ir.exists():
        print(f"V-JEPA IR not at {args.ir}", file=sys.stderr)
        return 1

    spec = PreprocessSpec.load_for_model(args.ir)

    config = IronConfig.load()
    version = args.set_version or config.eval.golden_set_version
    golden = load_golden_set(config.paths.resolve(config.eval.golden_sets_dir), version)
    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else "synthetic-indoor-v3"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    clips = list(golden.clips)
    if args.limit:
        clips = clips[: args.limit]

    # Gate first, per the Day-10 rule; report the appearance gate verdict on
    # the set so the reader knows what to expect from the metrics.
    gate_verdicts = []
    all_frames_sample = None
    for clip in clips[:8]:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        r = validity.evaluate(
            "appearance_semantics", clip.clip_id, frames=data["rgb"]
        )
        gate_verdicts.append({"clip_id": clip.clip_id, **r.as_dict()})
        state = "PASS" if r.passed else "REFUSE"
        te = r.evidence.get("texture_energy")
        md = r.evidence.get("material_diversity")
        print(f"{state}  appearance_semantics  {clip.clip_id}  texture={te} materials={md}")
        if all_frames_sample is None:
            all_frames_sample = data["rgb"]

    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(str(args.ir), "CPU")
    infer = compiled.create_infer_request()

    # Load clip data ONCE — every path shares it.
    clips_data = []
    for clip in clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        clips_data.append(
            {
                "clip_id": clip.clip_id,
                "rgb": data["rgb"],
                "track_uv": data["track_uv"],
                "track_occluded": data["track_occluded"],
            }
        )
    print(f"\nloaded {len(clips_data)} clips")

    visits = patch_visit_stats(clips_data, spec)
    print("\n---- patch-visit diagnostic (measuring apparatus check) ----")
    for k, v in visits.items():
        print(f"  {k}: {v}")

    print("\n---- standardised path (production) ----")
    fixed_metrics, _ = run_path(infer, spec, clips_data, standardise=True)
    for k, v in fixed_metrics.items():
        print(f"  {k}: {v}")

    print("\n---- pre-fix path (raw [0,1], reconstructed behind a flag) ----")
    prefix_metrics, _ = run_path(infer, spec, clips_data, standardise=False)
    for k, v in prefix_metrics.items():
        print(f"  {k}: {v}")

    def delta(a: float, b: float) -> float:
        return float(a - b)

    delta_report = {
        "mAP": delta(fixed_metrics["mAP"], prefix_metrics["mAP"]),
        "temporal_cosine": delta(
            fixed_metrics["temporal_cosine"], prefix_metrics["temporal_cosine"]
        ),
        "patch_boundary_l2": delta(
            fixed_metrics["patch_boundary_l2"], prefix_metrics["patch_boundary_l2"]
        ),
    }
    print("\n---- normalization delta (fixed - prefix) ----")
    for k, v in delta_report.items():
        print(f"  {k}: {v}")

    payload = {
        "set": version,
        "dataset": dataset,
        "spec_source": spec.source,
        "appearance_gate_sample": gate_verdicts,
        "patch_visit_diagnostic": visits,
        "standardised_metrics": {k: (None if not np.isfinite(v) else v) for k, v in fixed_metrics.items()},
        "prefix_metrics": {k: (None if not np.isfinite(v) else v) for k, v in prefix_metrics.items()},
        "delta_fixed_minus_prefix": delta_report,
        "interpretation_note": (
            "The appearance gate refuses v3 on measured evidence (texture "
            "energy 3.03, floor 12.0). Read the patch_visit_diagnostic FIRST: "
            "if most tracks occupy a single patch over the encoder window, "
            "mAP is dominated by patch-location constancy rather than by V-JEPA "
            "understanding of the agent as an object, so a high mAP is NOT "
            "evidence against the appearance gate — it's the metric measuring "
            "the wrong thing. A pooled score across a fixture that cannot make "
            "tracks traverse patches within the encoder's 4-frame window is a "
            "same-frame lookup dressed as retrieval. The normalization delta "
            "should be trusted more than mAP here: standardisation is what "
            "makes V-JEPA embeddings semantically meaningful, so if the delta "
            "is near zero the mAP is not doing semantic work at all."
        ),
        "checkpoint_note": (
            "V-JEPA2 fp32 IR at models/export/2026-07-31/. No INT8 (blocked "
            "on the production artifact human item). All numbers above are "
            "from fp32."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
