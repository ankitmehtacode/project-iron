"""Score V-JEPA2 semantics against v3-indoor's GT tracks, gate first.

Day 11 repair, and why it was needed
-------------------------------------
Day 11's first version of this script queried at frame 0 and pooled every
frame of a 4-frame clip (~0.13 s of footage). The resulting mAP (0.268,
~18x chance) looked like evidence against the appearance-gate partition.
The diagnostic that shipped alongside it — ``patch_visit_diagnostic`` —
showed why that was wrong: 79% of GT tracks never leave their 16-px patch
across that window, so "same-object retrieval" reduced to "look up almost
the same embedding four times." Position, not appearance, was doing the
work, and a query/pool protocol that cannot make tracks traverse patches
cannot distinguish the two.

This version repairs the protocol structurally rather than patching the
symptom:

- **Cross-boundary queries only.** A (query, target) pair at gap ``k``
  frames apart is scored only if the GT track's patch index changed at
  least once between frame 0 and frame ``k`` — computed from GT position
  directly, no encoder call needed for the filter itself. Pairs that never
  cross a boundary are dropped, and the surviving count is reported PER
  GAP: if it is tiny, that is the finding, not a footnote.
- **Temporal gap is an explicit, swept parameter** (``--gaps``, default
  "1,2,4,8,16,32" frames, filtered to what a 40-frame clip supports). Each
  gap gets its own encoder passes at offset 0 and at offset ``k``, its own
  surviving-pair count, its own mAP.
- **Distractor design.** The retrieval pool at gap ``k`` is every GT
  track's embedding at frame ``k`` — every agent in this dataset is the
  same class (person), so the pool is same-class/different-instance by
  construction; retrieving "a person" is not this task, retrieving
  "the SAME person" is, because every pool member is a candidate for every
  query's positive match by track identity alone.
- **Position-only baseline, computed and reported at every gap.** Ranks
  the pool by proximity in patch-grid coordinates between the query's
  frame-0 position and each candidate's frame-``k`` position — the exact
  "assume it barely moved" strategy that explained Day 11's number. If
  this baseline is itself well above chance on the cross-boundary set,
  that is worth knowing; if it collapses to chance while the encoder's
  mAP does not, that is the first real evidence for or against the
  partition this project has produced.

Window length (the number of frames V-JEPA2 consumes per call, ``T`` in
ADR 0002) is a separate axis from temporal gap. The only IR exported so
far is fixed at ``T=4`` (see ``models/export/2026-07-31/…preprocess.json``,
``frames: 4``); ADR 0002 needed a semantic metric to decide 16-vs-64 and
none existed. This script is that metric NOW — the gap-sweep and
cross-boundary protocol above generalise directly to any exported window
length — but running it at ``T=16`` or ``T=64`` needs a new export that
does not exist today. ``--window`` is exposed and refuses cleanly for any
value other than the one the loaded IR was exported for, rather than
silently ignoring the request or crashing inside OpenVINO with an opaque
shape error.

Every metric is computed twice per gap: once with the sidecar
preprocessing spec applied (production path) and once with the raw
[0, 1] clip fed straight in (pre-fix path, reconstructed behind a flag;
production preprocessing is not un-fixed anywhere).

    python scripts/eval_semantics.py
    python scripts/eval_semantics.py --gaps 1,4,16 --limit 8

Exit codes:
    0  gates ran and metrics were computed at every gap.
    1  weights, IR, spec, or clips missing.
    2  --window does not match the loaded IR's exported frame count.
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
from src.eval.baselines import Baseline, margin as compute_margin  # noqa: E402
from src.models.preprocess import PreprocessSpec  # noqa: E402
from src.semantics.patch_mapping import (  # noqa: E402
    map_tracks_to_embeddings,
    tokens_from_encoder_output,
)

DEFAULT_IR = Path("models/export/2026-07-31/vjepa2_vitl_fp32.xml")
DEFAULT_GAPS = "1,2,4,8,16,32"


def resize_and_scale_clip(rgb_uint8: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Resize a window of frames to the spec's resolution, scale to [0, 1].

    Returned shape ``[1, T, 3, H, W]``, ``T == rgb_uint8.shape[0]``.
    Deliberately does NOT standardise — standardisation is what gets
    ablated by the caller.
    """
    import cv2

    height, width = spec.resolution
    resized = np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in rgb_uint8
        ]
    )
    scaled = resized.astype(np.float32) / 255.0
    return scaled.transpose(0, 3, 1, 2)[None]


def encode(
    infer: Any, clip_bt_chw: np.ndarray, spec: PreprocessSpec, standardise: bool
) -> np.ndarray:
    """Run V-JEPA2 IR on one window, honouring or bypassing the standardiser."""
    prepped = spec.apply(clip_bt_chw) if standardise else clip_bt_chw.astype(np.float32)
    infer.infer({"video": prepped})
    return infer.get_output_tensor(0).data.copy()


def patch_index(
    uv: np.ndarray, spec: PreprocessSpec, orig_h: int, orig_w: int
) -> np.ndarray:
    """``(row, col)`` on the encoder's patch grid for pixel coordinates ``uv``.

    ``uv`` is ``[..., 2]`` in the ORIGINAL raster; scaled to the encoder's
    input resolution before dividing by patch size, so this matches the
    geometry the encoder actually sees — the same scaling
    ``embed_tracks_at_offset`` applies to the tracks it feeds the patch
    lookup.
    """
    encoder_h, encoder_w = spec.resolution
    sx, sy = encoder_w / orig_w, encoder_h / orig_h
    scaled_x = uv[..., 0] * sx
    scaled_y = uv[..., 1] * sy
    row = np.floor(scaled_y / spec.patch_size).astype(np.int32)
    col = np.floor(scaled_x / spec.patch_size).astype(np.int32)
    return np.stack([row, col], axis=-1)


def observable_at(
    gt_uv_track: np.ndarray,
    gt_occ_track: np.ndarray,
    frame: int,
    orig_h: int,
    orig_w: int,
) -> bool:
    """Is this ONE track in-frame and unoccluded at ``frame``?

    ``gt_uv_track`` is ``[T, 2]`` and ``gt_occ_track`` is ``[T]`` — already
    indexed down to a single track by the caller.
    """
    x, y = gt_uv_track[frame]
    in_frame = 0 <= x < orig_w and 0 <= y < orig_h
    return bool(in_frame and not gt_occ_track[frame])


def crossed_boundary(
    gt_uv_track: np.ndarray,
    spec: PreprocessSpec,
    orig_h: int,
    orig_w: int,
    frame0: int,
    framek: int,
) -> bool:
    """Did this track's patch index change at least once in ``[frame0, framek]``?

    ``gt_uv_track`` is ``[T, 2]`` for a single track. Computed entirely from
    GT position — no encoder call. This is the filter that makes retrieval
    impossible to solve by position alone: a query/target pair that never
    crosses a patch boundary is exactly the Day-11 failure mode, and it is
    excluded here rather than diluting the pool.
    """
    span = gt_uv_track[frame0 : framek + 1]  # [k+1, 2]
    patches = patch_index(span, spec, orig_h, orig_w)  # [k+1, 2]
    base = patches[0]
    return bool(np.any(np.any(patches != base, axis=-1)))


def embed_tracks_at_offset(
    infer: Any,
    spec: PreprocessSpec,
    rgb: np.ndarray,
    gt_uv: np.ndarray,
    offset: int,
    *,
    standardise: bool,
) -> np.ndarray | None:
    """Every track's embedding at frame ``offset``. Shape ``[N, dim]``.

    Encodes the window ``rgb[offset : offset + spec.frames]`` and reads out
    slot 0 (frame ``offset`` of the clip, frame 0 of the window). Returns
    ``None`` when the window does not fit inside the clip.
    """
    if offset + spec.frames > rgb.shape[0]:
        return None
    orig_h, orig_w = rgb.shape[1], rgb.shape[2]
    encoder_h, encoder_w = spec.resolution
    sx, sy = encoder_w / orig_w, encoder_h / orig_h

    window = resize_and_scale_clip(rgb[offset : offset + spec.frames], spec)
    features = encode(infer, window, spec, standardise=standardise)

    grid = (encoder_h // spec.patch_size, encoder_w // spec.patch_size)
    # No wall clock: this is encoder output, not a captured interval.
    span = TemporalSpan.without_wall_clock(
        frames_covered=spec.frames, tubelet=spec.tubelet
    )
    tokens = tokens_from_encoder_output(
        features,
        batch_index=0,
        span=span,
        grid=grid,
        geometry=FrameGeometry(width=encoder_w, height=encoder_h),
        encoder_sha="ablation",
    )
    tracks_window = gt_uv[offset : offset + spec.frames].copy().astype(np.float32)
    tracks_window[..., 0] *= sx
    tracks_window[..., 1] *= sy
    embeddings = map_tracks_to_embeddings(tokens, tracks_window)  # [T, N, dim]
    return embeddings[0]  # frame `offset` == window frame 0 == slot 0


def average_precision(
    query_idx: int, sim: np.ndarray, labels: np.ndarray, query_label: int
) -> float | None:
    """AP for one query against a ranked candidate pool.

    ``sim`` is the query's similarity to every pool member (higher = closer
    rank). Returns ``None`` when the query has no true positive in the pool
    — should not happen by construction (the query's own track always has
    an embedding in the same-gap pool) but guarded rather than assumed.
    """
    order = np.argsort(-sim)
    ranked_labels = labels[order]
    hits = ranked_labels == query_label
    n_pos = int(hits.sum())
    if n_pos == 0:
        return None
    precisions = np.cumsum(hits, dtype=np.float64) / (np.arange(len(hits)) + 1)
    return float((precisions * hits).sum() / n_pos)


def patch_visit_stats(
    clips_data: list[dict[str, Any]], spec: PreprocessSpec
) -> dict[str, Any]:
    """Day-11 diagnostic, kept: how many patches a track visits in ONE window.

    Retained for continuity with the Day-11 finding this script repairs —
    the number here (still computed over the single 4-frame window at
    offset 0) is what explained the original mAP; the gap-swept metrics
    below are the actual repair.
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
        patches = np.floor(scaled / patch).astype(int)
        for n in range(patches.shape[1]):
            uniq = len({tuple(p) for p in patches[:, n]})
            per_track_unique.append(uniq)
    arr = np.asarray(per_track_unique)
    return {
        "n_tracks": int(arr.size),
        "unique_patches_mean": float(arr.mean()) if arr.size else float("nan"),
        "unique_patches_max": int(arr.max()) if arr.size else 0,
        "fraction_in_one_patch": float((arr == 1).mean()) if arr.size else float("nan"),
        "fraction_in_at_most_two_patches": float((arr <= 2).mean())
        if arr.size
        else float("nan"),
    }


def run_gap_sweep(
    infer: Any,
    spec: PreprocessSpec,
    clips_data: list[dict[str, Any]],
    gaps: list[int],
    *,
    standardise: bool,
) -> dict[int, dict[str, Any]]:
    """Cross-boundary retrieval mAP, position-only baseline, per gap.

    Encodes every offset in ``{0} | gaps`` once per clip (shared across all
    gaps that reference it — offset 0 is reused by every gap), builds a
    global track-id space, and for each gap constructs the query set
    (cross-boundary, observable at both ends) and the pool (every track
    observable at frame ``k``).
    """
    offsets = sorted({0, *gaps})

    # embeddings[offset] -> list of (global_id, embedding) for every track
    # observable and encodable at that offset.
    embeddings_by_offset: dict[int, list[tuple[int, np.ndarray]]] = {
        o: [] for o in offsets
    }
    # positions[(clip_id, track)] -> gt_uv, gt_occ arrays, kept for the
    # boundary-crossing and position-only-baseline computations.
    track_meta: dict[tuple[str, int], dict[str, Any]] = {}

    global_id = 0
    for clip in clips_data:
        rgb, gt_uv, gt_occ = clip["rgb"], clip["track_uv"], clip["track_occluded"]
        orig_h, orig_w = rgb.shape[1], rgb.shape[2]
        n_tracks = gt_uv.shape[1]

        per_offset_emb: dict[int, np.ndarray | None] = {}
        for offset in offsets:
            per_offset_emb[offset] = embed_tracks_at_offset(
                infer, spec, rgb, gt_uv, offset, standardise=standardise
            )

        for track in range(n_tracks):
            gid = global_id
            global_id += 1
            gt_uv_track, gt_occ_track = gt_uv[:, track], gt_occ[:, track]
            track_meta[(clip["clip_id"], track)] = {
                "gid": gid,
                "gt_uv": gt_uv_track,
                "gt_occ": gt_occ_track,
                "orig_h": orig_h,
                "orig_w": orig_w,
            }
            for offset in offsets:
                emb_at_offset = per_offset_emb[offset]
                if emb_at_offset is None:
                    continue
                if not observable_at(gt_uv_track, gt_occ_track, offset, orig_h, orig_w):
                    continue
                embeddings_by_offset[offset].append((gid, emb_at_offset[track]))

    # Reverse lookup: global id -> its track_meta entry, and its offset-0
    # embedding, both built once rather than re-scanned per gap.
    gid_to_meta = {m["gid"]: m for m in track_meta.values()}
    gid_to_zero_embedding = {g: e for g, e in embeddings_by_offset[0]}

    results: dict[int, dict[str, Any]] = {}
    for k in gaps:
        if not embeddings_by_offset.get(k) or not embeddings_by_offset[0]:
            results[k] = {"n_pairs": 0, "note": "offset does not fit any clip"}
            continue

        pool_gids = np.array([g for g, _ in embeddings_by_offset[k]])
        pool_embs = np.stack([e for _, e in embeddings_by_offset[k]])
        pool_gid_set = set(int(g) for g in pool_gids)

        query_rows = []  # (gid, embedding, patch0)
        for gid in pool_gid_set:
            if gid not in gid_to_zero_embedding:
                continue  # not observable/encodable at offset 0
            meta = gid_to_meta[gid]
            gt_uv_track, orig_h, orig_w = meta["gt_uv"], meta["orig_h"], meta["orig_w"]
            if gt_uv_track.shape[0] <= k:
                continue
            if not crossed_boundary(gt_uv_track, spec, orig_h, orig_w, 0, k):
                continue
            patch0 = patch_index(gt_uv_track[0], spec, orig_h, orig_w)
            query_rows.append((gid, gid_to_zero_embedding[gid], patch0))

        if not query_rows:
            results[k] = {
                "n_pairs": 0,
                "n_pool": int(pool_embs.shape[0]),
                "note": "no track crossed a patch boundary between frame 0 and this gap",
            }
            continue

        query_gids = np.array([q[0] for q in query_rows])
        query_embs = np.stack([q[1] for q in query_rows])
        query_patch0 = np.stack([q[2] for q in query_rows])  # [Q, 2]

        # Embedding-based retrieval.
        sim_matrix = query_embs @ pool_embs.T  # [Q, P] cosine (L2-normalised)
        aps = []
        for i in range(len(query_rows)):
            ap = average_precision(i, sim_matrix[i], pool_gids, query_gids[i])
            if ap is not None:
                aps.append(ap)
        map_score = float(np.mean(aps)) if aps else float("nan")

        # Position-only baseline: rank the pool by proximity to the query's
        # OWN frame-0 patch index — the "assume it barely moved" strategy.
        pool_patches_k = np.stack(
            [
                patch_index(
                    gid_to_meta[int(gid)]["gt_uv"][k],
                    spec,
                    gid_to_meta[int(gid)]["orig_h"],
                    gid_to_meta[int(gid)]["orig_w"],
                )
                for gid in pool_gids
            ]
        )  # [P, 2]

        pos_aps = []
        for i in range(len(query_rows)):
            dist = -np.linalg.norm(pool_patches_k - query_patch0[i], axis=1)
            ap = average_precision(i, dist, pool_gids, query_gids[i])
            if ap is not None:
                pos_aps.append(ap)
        position_only_map = float(np.mean(pos_aps)) if pos_aps else float("nan")

        chance_map = 1.0 / pool_embs.shape[0] if pool_embs.shape[0] else float("nan")

        results[k] = {
            "n_pairs": len(query_rows),
            "n_pool": int(pool_embs.shape[0]),
            "mAP": map_score,
            "position_only_mAP": position_only_map,
            "chance_mAP": chance_map,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--set", dest="set_version", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gaps", default=DEFAULT_GAPS)
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="V-JEPA clip length T. Must match the loaded IR's export; "
        "sweeping T itself needs a new export (see ADR 0002).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/day12/semantics.json")
    )
    args = parser.parse_args(argv)

    if not args.ir.exists():
        print(f"V-JEPA IR not at {args.ir}", file=sys.stderr)
        return 1

    spec = PreprocessSpec.load_for_model(args.ir)

    if args.window is not None and args.window != spec.frames:
        print(
            f"--window {args.window} does not match this IR's exported frame "
            f"count ({spec.frames}). Window length (T in ADR 0002) is fixed "
            "at export time; sweeping it requires exporting a new IR at the "
            "requested T, which does not exist yet. Refusing rather than "
            "feeding a mismatched shape into OpenVINO or silently ignoring "
            "the request.",
            file=sys.stderr,
        )
        return 2

    gaps = sorted({int(g) for g in args.gaps.split(",") if g.strip()})

    config = IronConfig.load()
    version = args.set_version or config.eval.golden_set_version
    golden = load_golden_set(config.paths.resolve(config.eval.golden_sets_dir), version)
    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else "synthetic-indoor-v3"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    clips = list(golden.clips)
    if args.limit:
        clips = clips[: args.limit]

    gate_verdicts = []
    for clip in clips[:8]:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        r = validity.evaluate("appearance_semantics", clip.clip_id, frames=data["rgb"])
        gate_verdicts.append({"clip_id": clip.clip_id, **r.as_dict()})
        state = "PASS" if r.passed else "REFUSE"
        print(f"{state}  appearance_semantics  {clip.clip_id}")

    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(str(args.ir), "CPU")
    infer = compiled.create_infer_request()

    clips_data = []
    max_gap = max(gaps) if gaps else 0
    for clip in clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if data["rgb"].shape[0] < spec.frames + max_gap:
            continue  # this clip is too short for the largest gap requested
        clips_data.append(
            {
                "clip_id": clip.clip_id,
                "rgb": data["rgb"],
                "track_uv": data["track_uv"],
                "track_occluded": data["track_occluded"],
            }
        )
    print(f"\nloaded {len(clips_data)} clips (window T={spec.frames}, gaps={gaps})")

    visits = patch_visit_stats(clips_data, spec)
    print("\n---- patch-visit diagnostic (single-window, Day-11 finding) ----")
    for k, v in visits.items():
        print(f"  {k}: {v}")

    print("\n---- standardised path (production) ----")
    fixed = run_gap_sweep(infer, spec, clips_data, gaps, standardise=True)
    print("\n---- pre-fix path (raw [0,1], reconstructed behind a flag) ----")
    prefix = run_gap_sweep(infer, spec, clips_data, gaps, standardise=False)

    per_gap_report = {}
    print("\n---- per-gap results ----")
    print(
        f"{'gap':>5} {'pairs':>6} {'pool':>6} {'mAP':>8} {'baseline':<24} {'margin':>9}  {'delta(fix-prefix)':>18}"
    )
    for k in gaps:
        f = fixed.get(k, {})
        p = prefix.get(k, {})
        n_pairs = f.get("n_pairs", 0)
        if n_pairs == 0:
            print(
                f"{k:>5} {0:>6}   -- no surviving cross-boundary pairs --  ({f.get('note', '')})"
            )
            per_gap_report[k] = {"standardised": f, "prefix": p}
            continue

        chance_b = Baseline(
            "chance", f["chance_mAP"], "1 / pool size", flag_worthy=True
        )
        pos_b = Baseline(
            "position_only",
            f["position_only_mAP"],
            "rank by patch-grid proximity between query frame-0 position and "
            "candidate frame-k position — ignores the encoder entirely",
            flag_worthy=True,
        )
        m = compute_margin(f["mAP"], [chance_b, pos_b], higher_is_better=True)
        strongest = max(f["chance_mAP"], f["position_only_mAP"])
        delta_map = f["mAP"] - p.get("mAP", float("nan"))

        flag = "  !! FLAGGED" if np.isfinite(m) and m <= 0 else ""
        print(
            f"{k:>5} {n_pairs:>6} {f['n_pool']:>6} {f['mAP']:>8.4f} "
            f"strongest: {strongest:>8.4f}   {m:>+9.4f}  {delta_map:>+18.4f}{flag}"
        )
        per_gap_report[k] = {
            "standardised": f,
            "prefix": p,
            "delta_mAP": delta_map,
            "margin_over_strongest_baseline": m,
            "flagged": bool(np.isfinite(m) and m <= 0),
        }

    payload = {
        "set": version,
        "dataset": dataset,
        "window_T": spec.frames,
        "window_note": (
            "T is fixed by the exported IR (4 frames). ADR 0002's 16-vs-64 "
            "decision needed a semantic metric to decide it; this script's "
            "gap-sweep + cross-boundary protocol is that metric, but no "
            "IR has been exported at T=16 or T=64 yet, so no data exists "
            "at those windows. No clip-length policy is adopted today."
        ),
        "gaps_swept": gaps,
        "appearance_gate_sample": gate_verdicts,
        "patch_visit_diagnostic": visits,
        "per_gap": {str(k): v for k, v in per_gap_report.items()},
        "interpretation_note": (
            "Read n_pairs before mAP. A gap where n_pairs is tiny relative "
            "to the track count means almost no track crosses a patch "
            "boundary in that many frames — the fixture's motion is too "
            "slow, not the encoder's fault, and the mAP at that gap should "
            "not be trusted. Compare mAP against BOTH baselines: chance "
            "(uniform-random ranking) and position_only (rank by assumed "
            "co-location). If mAP is close to position_only, the encoder is "
            "not adding information beyond spatial proximity — the same "
            "failure mode Day 11 found, just no longer hidden by a query "
            "protocol that could not detect it."
        ),
        "checkpoint_note": (
            "V-JEPA2 fp32 IR at models/export/2026-07-31/. No INT8 (blocked "
            "on the production artifact human item)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
