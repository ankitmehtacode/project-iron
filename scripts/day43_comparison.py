"""Day 43: one table, oracle vs pretrained, from the runs' own eval JSONs.

Per camera and pooled: event precision/recall (estimator and raw
projection), ID switches (pipeline vs IoU-only baseline), and, by camera
distance bucket, position RMSE, velocity RMSE (both estimator vs raw
projection) and coverage. Nothing is recomputed: every number is read from
``<run>/eval/*.json`` (the newest), so the table cannot disagree with the
evaluator.

    python scripts/day43_comparison.py \
        --run oracle=outputs/runs/.../oracle_gate-off \
        --run pretrained=outputs/runs/.../rtdetr_v2_s_gate-off \
        --out artifacts/day43/x.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def latest_eval(run: Path) -> dict[str, Any]:
    files = sorted((run / "eval").glob("*.json"))
    if not files:
        raise SystemExit(f"{run} has no eval/*.json; run `python -m iron eval` first")
    data: dict[str, Any] = json.loads(files[-1].read_text())
    data["_file"] = str(files[-1])
    return data


def f3(x: Any) -> str:
    return "—" if x is None else f"{x:.3f}"


def block_rows(name: str, scope: str, b: dict[str, Any]) -> list[str]:
    ev = b["events"]
    est, raw = ev["estimator"], ev["raw_projection"]
    mot = b["mot"]
    head = (
        f"| {name} | {scope} | {f3(est['precision'])} / {f3(est['recall'])} "
        f"({est['matched']}/{est['predicted']}/{est['gt']}) | "
        f"{f3(raw['precision'])} / {f3(raw['recall'])} | "
        f"{mot['pipeline']['id_switches']} / "
        f"{mot['iou_only_baseline']['id_switches']} |"
    )
    return [head]


def bucket_rows(name: str, scope: str, b: dict[str, Any]) -> list[str]:
    out = []
    pos = b["rmse_m_by_distance"]
    vel = b.get("velocity_rmse_mps_by_distance") or {}
    cov = b["coverage_by_distance"]
    for bucket in sorted(cov, key=lambda k: float(k[1:].split(",")[0])):
        p, v, c = pos.get(bucket, {}), vel.get(bucket, {}), cov[bucket]
        out.append(
            f"| {name} | {scope} | {bucket} | {f3(p.get('estimator'))} / "
            f"{f3(p.get('raw_projection'))} | {f3(p.get('estimator_p50'))} / "
            f"{f3(p.get('raw_projection_p50'))} | {f3(v.get('estimator'))} / "
            f"{f3(v.get('raw_projection'))} | {f3(c['coverage'])} "
            f"({c['matched']}/{c['gt']}) |"
        )
    return out


def render(runs: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "*Lane S synthetic (NVIDIA PhysicalAI-SmartSpaces Warehouse_022), "
        "calibration given; not evidence about real cameras.*",
        "",
        "Sources: " + "; ".join(f"{n} = `{e['_file']}`" for n, e in runs),
        "",
        "## Events and identity",
        "",
        "| detector | scope | events, estimator P / R (matched/pred/GT) | "
        "events, raw projection P / R | ID switches, pipeline / IoU-only |",
        "|---|---|---|---|---|",
    ]
    for name, e in runs:
        for cam, b in e["cameras"].items():
            lines += block_rows(name, cam, b)
        lines += block_rows(name, "**pooled**", e["pooled"])
    lines += [
        "",
        "## Position, velocity and coverage by camera distance",
        "",
        "| detector | scope | bucket (m) | position RMSE est / raw (m) | "
        "position p50 est / raw | velocity RMSE est / raw (m/s) | coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, e in runs:
        lines += bucket_rows(name, "**pooled**", e["pooled"])
    for name, e in runs:
        for cam, b in e["cameras"].items():
            lines += bucket_rows(name, cam, b)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", action="append", required=True, help="name=RUN_DIR")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs.append((name, latest_eval(Path(path))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(runs))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
