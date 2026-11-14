"""Day 46, Objective 2: apply rules R1-R6 to the RECORDED Day-44 results.

No new measurement: every number is read from a Day-44 artifact, and each
rule's text is copied from the Day-46 prompt, where it was fixed before the
numbers were looked at. Writes ``artifacts/day46/rules.json``.

    python scripts/day46_rules.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DAY44 = ROOT / "artifacts" / "day44"
OUT = ROOT / "artifacts" / "day46" / "rules.json"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DAY44 / name).read_text())
    return data


def _overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def r1_schedule() -> dict[str, Any]:
    src = "frontier_w022_pretrained_iou.json"
    cands = _load(src)["candidates"]
    ref = cands["gate_off"]["pooled"]["bootstrap95_over_gt_tracks"]["recall"]
    rows = []
    for name, c in cands.items():
        if name == "gate_off":
            continue
        p = c["pooled"]
        rec = p["bootstrap95_over_gt_tracks"]["recall"]
        rows.append(
            {
                "candidate": name,
                "invocations": p["invocations"],
                "recall_ci95": rec,
                "overlaps_gate_off": _overlaps(rec, ref),
            }
        )
    passing = sorted(
        (r for r in rows if r["overlaps_gate_off"]), key=lambda r: r["invocations"]
    )
    chosen = passing[0]["candidate"] if passing else "stride_6"
    table = (DAY44 / "frontier_tables_pretrained_iou.md").read_text()
    paired = next(
        line.split("|")[5].strip()
        for line in table.splitlines()
        if line.startswith(f"| {chosen} |")
    )
    return {
        "rule": (
            "pretrained detections, IoU-only tracker: the candidate with the "
            "fewest invocations whose event-recall interval overlaps gate-off's; "
            "if none, stride 6"
        ),
        "source": f"artifacts/day44/{src}",
        "gate_off_recall_ci95": ref,
        "candidates": rows,
        "passing": [r["candidate"] for r in passing],
        "outcome": chosen,
        "paired_vs_gate_off": paired,
    }


def r2_tracker() -> dict[str, Any]:
    out: dict[str, Any] = {
        "rule": (
            "iron run defaults to whichever tracker has fewer ID switches on "
            "pretrained detections"
        )
    }
    switches = {}
    for tracker in ("iou", "kalman"):
        src = f"frontier_w022_pretrained_{tracker}.json"
        cands = _load(src)["candidates"]
        switches[tracker] = {
            sched: sum(
                cam["mot"]["pipeline"]["id_switches"]
                for cam in cands[sched]["cameras"].values()
            )
            for sched in ("gate_off", "adaptive")
        }
        out[f"source_{tracker}"] = f"artifacts/day44/{src}"
    out["id_switches"] = switches
    better = {
        sched: min(switches, key=lambda t: switches[t][sched])
        for sched in ("gate_off", "adaptive")
    }
    out["fewer_by_schedule"] = better
    out["outcome"] = better["gate_off"] if len(set(better.values())) == 1 else None
    return out


def r3_range_interval() -> dict[str, Any]:
    src = "range_interval_oracle.json"
    scenes = _load(src)["scenes"]
    rows = {}
    for scene, block in scenes.items():
        b = block["population_bound"]
        rows[scene] = {
            "boxes": b["boxes"],
            "inside_true_range": b["inside"],
            "coverage_true_range": b["coverage"],
            "coverage_near_edge_reference": b["coverage_near_edge_reference"],
            "violations_upper_end": b["violations_upper_end"],
            "violations_lower_end": b["violations_lower_end"],
        }
    passes = all(r["coverage_true_range"] >= 0.95 for r in rows.values())
    return {
        "rule": (
            "adopt [R_head, R_foot] as the single-camera observation only if the "
            "true range lies inside it for >= 95% of person detections on EACH of "
            "the three scenes (oracle boxes); otherwise keep the foot anchor and "
            "list the violations by cause"
        ),
        "source": f"artifacts/day44/{src} (population_bound, H_max 2.058 m)",
        "true_range_definition": (
            "the artifact's own: GT footprint CENTRE (its reference_note); the "
            "near-edge figure is reported beside it, not used"
        ),
        "scenes": rows,
        "outcome": "adopt_interval" if passes else "keep_foot_anchor",
    }


def r4_uncertainty() -> dict[str, Any]:
    src = "error_model.json"
    held = _load(src)["heldout"]
    rows = {}
    for scene in ("Warehouse_020", "Warehouse_022"):
        h = held[scene]
        row = {}
        for level, nominal in (("68", 0.68), ("95", 0.95)):
            fit, const = h[f"coverage{level}_fitted"], h[f"coverage{level}_constant"]
            row[level] = {
                "fitted": fit,
                "constant": const,
                "fitted_closer": abs(fit - nominal) < abs(const - nominal),
            }
        rows[scene] = row
    passes = all(lv["fitted_closer"] for r in rows.values() for lv in r.values())
    return {
        "rule": (
            "the fitted error model replaces the constant only if its held-out "
            "coverage is closer to nominal than the constant's on every held-out "
            "scene (applied at both nominal levels, 68% and 95%)"
        ),
        "source": f"artifacts/day44/{src}",
        "heldout": rows,
        "outcome": "fitted_model" if passes else "constant",
    }


def r5_position() -> dict[str, Any]:
    src = "frontier_w022_pretrained_iou.json"
    pooled = _load(src)["candidates"]["gate_off"]
    buckets: dict[str, Any] = {}
    for cam in pooled["cameras"].values():
        for b, row in cam["rmse_m_by_distance"].items():
            acc = buckets.setdefault(b, {"n": 0, "est_sq": 0.0, "raw_sq": 0.0})
            if row["n"]:
                acc["n"] += row["n"]
                acc["est_sq"] += row["estimator"] ** 2 * row["n"]
                acc["raw_sq"] += row["raw_projection"] ** 2 * row["n"]
    rows = {
        b: {
            "n": a["n"],
            "estimator": (a["est_sq"] / a["n"]) ** 0.5,
            "raw": (a["raw_sq"] / a["n"]) ** 0.5,
        }
        for b, a in buckets.items()
        if a["n"]
    }
    worse = [b for b, r in rows.items() if r["estimator"] > r["raw"]]
    return {
        "rule": (
            "if the estimator's position RMSE on pretrained detections is higher "
            "than raw projection in most distance buckets, raw (or fused) position "
            "is reported; the estimator supplies velocity, uncertainty, coasting"
        ),
        "source": f"artifacts/day44/{src} (gate_off, pooled over cameras by n)",
        "buckets": rows,
        "estimator_worse_in": worse,
        "outcome": "raw" if len(worse) > len(rows) / 2 else "estimator",
    }


def r6_fusion() -> dict[str, Any]:
    src = "multiview_ceiling_oracle.json"
    scenes = _load(src)["scenes"]
    rows = {}
    for scene, block in scenes.items():
        fusion = block["position_error"]["interval_fusion"]
        strata = {k: v for k, v in fusion.items() if k in ("2", "3+") and v["n"]}
        rows[scene] = {
            "strata_p90_m": {k: v["p90_m"] for k, v in strata.items()},
            "n": {k: v["n"] for k, v in strata.items()},
            # the p90 of a mixture is at most the largest stratum p90
            "p90_two_or_more_at_most_m": max(v["p90_m"] for v in strata.values()),
        }
    passes = all(r["p90_two_or_more_at_most_m"] <= 1.0 for r in rows.values())
    return {
        "rule": (
            "build Objective 3 only if the Day-44 ceiling (views paired by GT "
            "identity) gives p90 position error <= 1.0 m with two or more views "
            "on BOTH crowded scenes"
        ),
        "source": f"artifacts/day44/{src} (interval_fusion)",
        "scenes": rows,
        "outcome": "build" if passes else "skip",
    }


def main() -> int:
    result = {
        "label": "Day-44 recorded results, rules fixed in the Day-46 prompt",
        "R1": r1_schedule(),
        "R2": r2_tracker(),
        "R3": r3_range_interval(),
        "R4": r4_uncertainty(),
        "R5": r5_position(),
        "R6": r6_fusion(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for key in ("R1", "R2", "R3", "R4", "R5", "R6"):
        print(key, result[key]["outcome"])
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
