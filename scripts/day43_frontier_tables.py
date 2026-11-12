"""Day 43: frontier tables and the stride-15 event audit, from frontier JSONs.

Reads ``artifacts/day43/frontier_*.json`` (scripts/cascade_frontier.py) and
writes one markdown file. Nothing is recomputed except pairing:

- Event precision and recall as counts (matched/predicted, matched/GT).
- Paired difference against gate-off: GT events are identical across
  candidates (they depend only on the ground truth and the zones; checked),
  so each GT event is matched or not under both. Reported: events gained,
  events lost, and the recall difference with a 95% paired percentile
  bootstrap over GT person tracks (same units, same seed as the frontier).
- ID switches, pipeline tracker and IoU-only baseline, with the frontier's
  bootstrap intervals.
- Position RMSE by camera-distance bucket, estimator / raw projection (n).

Stride-15 audit: every GT event gate-off matched and stride 15 did not is
classified as
  * elsewhere_in_time: stride 15 predicted an event of the same verb, zone
    and attributed GT subject, but only more than the tolerance away. The
    gap is reported; a gap of many seconds is most likely the same person's
    different visit, not a late copy of this event, so this bucket is NOT
    "the event fired late";
  * taken: such an event exists within tolerance but the greedy one-to-one
    matching gave it to another GT event;
  * lost, identity: no such event for that subject, but stride 15 did
    predict the same verb and zone within tolerance, attributed to another
    GT subject (or to none): the event fired on a fragmented track;
  * lost, absent: no event of that verb and zone within tolerance at all.
The tolerance is ``src.pipeline.evaluate.EVENT_TOLERANCE_S``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.evaluate import EVENT_TOLERANCE_S  # noqa: E402
from src.pipeline.frontier import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED  # noqa: E402

REF = "gate_off"


def f3(x: Any) -> str:
    return "—" if x is None else f"{x:.3f}"


def matched_gt(log: dict[str, Any]) -> set[int]:
    return {j for _, j in log["pairs"]}


def paired(result: dict[str, Any], name: str) -> dict[str, Any]:
    """Gained/lost GT events vs gate-off and a paired bootstrap of the
    recall difference over GT person tracks (camera, subject)."""
    units: dict[tuple[str, str], list[int]] = {}
    gained = lost = 0
    for cam, block in result[name]["cameras"].items():
        log, ref = block["event_log"], result[REF]["cameras"][cam]["event_log"]
        if log["gt"] != ref["gt"]:
            raise SystemExit(f"{cam}: GT events differ between {name} and {REF}")
        mine, theirs = matched_gt(log), matched_gt(ref)
        gained += len(mine - theirs)
        lost += len(theirs - mine)
        for j, (subject, *_rest) in enumerate(log["gt"]):
            u = units.setdefault((cam, subject), [0, 0, 0])
            u[0] += 1
            u[1] += j in mine
            u[2] += j in theirs
    a = np.array(list(units.values()), dtype=np.float64).reshape(-1, 3)
    out: dict[str, Any] = {"gained": gained, "lost": lost, "units": int(a.shape[0])}
    if a.shape[0] == 0 or a[:, 0].sum() == 0:
        out["delta_recall"] = None
        out["delta_recall_95"] = [None, None]
        return out
    out["delta_recall"] = float((a[:, 1].sum() - a[:, 2].sum()) / a[:, 0].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, a.shape[0], size=(BOOTSTRAP_ROUNDS, a.shape[0]))
    s = a[idx].sum(axis=1)
    ok = s[:, 0] > 0
    d = (s[ok, 1] - s[ok, 2]) / s[ok, 0]
    out["delta_recall_95"] = [
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    ]
    return out


def stride15_audit(result: dict[str, Any]) -> dict[str, Any]:
    tol = int(EVENT_TOLERANCE_S * 1e9)
    counts = {
        "elsewhere_in_time": 0,
        "taken": 0,
        "lost_identity": 0,
        "lost_absent": 0,
    }
    shifts_s: list[float] = []
    matched_shift_s: list[float] = []
    rows = []
    for cam, block in result["stride_15"]["cameras"].items():
        log = block["event_log"]
        ref = result[REF]["cameras"][cam]["event_log"]
        mine = matched_gt(log)
        for i, j in log["pairs"]:
            matched_shift_s.append((log["predicted"][i][4] - log["gt"][j][3]) / 1e9)
        for j in sorted(matched_gt(ref) - mine):
            subject, verb, zone, ts = log["gt"][j]
            same = [
                p
                for p in log["predicted"]
                if p[1] == subject and p[2] == verb and p[3] == zone
            ]
            gaps = [abs(p[4] - ts) for p in same]
            near_other = [
                p
                for p in log["predicted"]
                if p[2] == verb and p[3] == zone and abs(p[4] - ts) <= tol
            ]
            if not same:
                kind = "lost_identity" if near_other else "lost_absent"
            elif min(gaps) <= tol:
                kind = "taken"
            else:
                kind = "elsewhere_in_time"
                shifts_s.append(min(gaps) / 1e9)
            counts[kind] += 1
            rows.append(
                {
                    "camera": cam,
                    "subject": subject,
                    "verb": verb,
                    "zone": zone,
                    "gt_ts_s": ts / 1e9,
                    "kind": kind,
                    "nearest_same_event_gap_s": min(gaps) / 1e9 if gaps else None,
                }
            )
    return {
        "tolerance_s": EVENT_TOLERANCE_S,
        "counts": counts,
        "elsewhere_gaps_s": sorted(shifts_s),
        "matched_signed_shift_s_p50": (
            float(np.median(matched_shift_s)) if matched_shift_s else None
        ),
        "matched_abs_shift_s_p90": (
            float(np.percentile(np.abs(matched_shift_s), 90))
            if matched_shift_s
            else None
        ),
        "rows": rows,
    }


def scene_table(d: dict[str, Any]) -> list[str]:
    result = d["candidates"]
    buckets = sorted(
        {b for c in result.values() for b in c["pooled"]["rmse_m_by_distance"]},
        key=lambda k: float(k[1:].split(",")[0]),
    )
    head = (
        "| candidate | invocations (per cam-min) | precision | recall | "
        "vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | "
        "ID sw IoU-only [95%] | "
        + " | ".join(f"RMSE {b} est / raw (n)" for b in buckets)
        + " |"
    )
    lines = [head, "|" + "---|" * (7 + len(buckets))]
    for name, c in result.items():
        p = c["pooled"]
        ev, mot, bs = p["events"], p["mot"], p["bootstrap95_over_gt_tracks"]
        if name == REF:
            pair = "reference"
        else:
            pr = paired(result, name)
            lo, hi = pr["delta_recall_95"]
            pair = (
                f"+{pr['gained']} / -{pr['lost']}, {pr['delta_recall']:+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]"
            )
        rm = []
        for b in buckets:
            r = p["rmse_m_by_distance"].get(b)
            rm.append(
                "—"
                if r is None
                else f"{f3(r['estimator'])} / {f3(r['raw_projection'])} ({r['n']})"
            )
        per_min = p["invocations_per_camera_minute"]
        bsb = bs["id_switches_baseline"]
        lines.append(
            f"| {name} | {p['invocations']} ({per_min:.0f}) "
            f"| {ev['matched']}/{ev['predicted']} | {ev['matched']}/{ev['gt']} "
            f"| {pair} "
            f"| {mot['pipeline']['id_switches']} "
            f"[{bs['id_switches'][0]:.0f}, {bs['id_switches'][1]:.0f}] "
            f"| {mot['iou_only_baseline']['id_switches']} "
            f"[{bsb[0]:.0f}, {bsb[1]:.0f}] "
            "| " + " | ".join(rm) + " |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("frontiers", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args(argv)
    md = [
        "*Lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration given; "
        "not evidence about real cameras.*",
        "",
        f"Event matching tolerance: {EVENT_TOLERANCE_S} s "
        "(`src.pipeline.evaluate.EVENT_TOLERANCE_S`), same verb, zone and "
        "attributed GT subject, one-to-one greedy by time gap.",
    ]
    audits = {}
    for path in args.frontiers:
        d = json.loads(path.read_text())
        title = f"{d['scene']}, {d['detector']}, {len(d['cameras'])} cameras x "
        title += f"{d['frames_per_camera']} frames"
        md += ["", f"## {title}", "", f"Source: `{path}`", ""]
        md += scene_table(d)
        if "stride_15" in d["candidates"]:
            a = stride15_audit(d["candidates"])
            audits[str(path)] = a
            c = a["counts"]
            md += [
                "",
                f"Stride 15 vs gate-off, GT events gate-off matched and stride 15 "
                f"did not: lost {c['lost_identity'] + c['lost_absent']} "
                f"({c['lost_identity']} fired within tolerance on a track "
                f"attributed to another GT subject or none, "
                f"{c['lost_absent']} with no such event at all); "
                f"{c['elsewhere_in_time']} with the same person, zone and verb "
                f"only more than {a['tolerance_s']} s away (gaps s: "
                f"{', '.join(f'{g:.2f}' for g in a['elsewhere_gaps_s']) or 'none'}; "
                "a different visit is the likelier reading than a late event); "
                f"taken by another GT event {a['counts']['taken']}. Matched events: "
                f"signed time shift p50 {f3(a['matched_signed_shift_s_p50'])} s, "
                f"|shift| p90 {f3(a['matched_abs_shift_s_p90'])} s.",
            ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    if args.json_out:
        args.json_out.write_text(json.dumps(audits, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
