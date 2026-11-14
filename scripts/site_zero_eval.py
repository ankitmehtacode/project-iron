"""Score the site-zero-solo capture, or the synthetic stand-in, through ONE
function (src.eval.site_zero.run_site_zero). Day 46.

Real capture (when data/raw/site_zero_solo/ holds the clips, marks.csv,
notes.yaml and zones.yaml, the consent record is signed, and the calibration
page has written the floor file); run it through scripts/on_ac.py, it runs
RT-DETRv2-S on every frame the gate keeps:

    python scripts/on_ac.py -- python scripts/site_zero_eval.py \\
        --calibration configs/calibration/site_zero_solo/site_zero_cam.yaml \\
        --out outputs/site_zero/<name>

Synthetic stand-in (no real person, no real frame; the oracle detector over
the stand-in's known boxes, and the calibration written by the calibration
page's own save function from computed clicks):

    python scripts/site_zero_eval.py --synthetic --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.site_zero import run_site_zero  # noqa: E402

REAL_DATA = ROOT / "data" / "raw" / "site_zero_solo"


def synthetic_inputs(data_dir: Path) -> tuple[Path, Any, str, dict[str, Any]]:
    """Write the stand-in; calibrate from its clicks through the calibration
    page's save(); build the oracle over its boxes. Returns
    ``(calibration, detector_factory, person_label, known_answers)``."""
    import numpy as np

    from src.data.site_zero_synthetic import write_stand_in
    from src.detect.backends import GroundTruthBoxes, OracleDetector
    from src.detect.contract import ClassVocabulary
    from src.eval.site_zero import load_marks, load_notes
    from src.inspector.calibrate import save

    known = write_stand_in(data_dir)
    notes = load_notes(data_dir / "notes.yaml")
    marks = [
        {"name": n, "x_m": m["xy"][0], "y_m": m["xy"][1]}
        for n, m in load_marks(data_dir / "marks.csv").items()
        if m["role"] == "calib"
    ]
    calibration = data_dir / "calibration.yaml"
    clicks = json.loads((data_dir / "clicks.json").read_text())
    known["calibration"] = save(
        clicks, marks, notes["camera_id"], calibration, data_dir / "A.mp4", 0
    )
    truth = json.loads((data_dir / "truth.json").read_text())
    by_file = {c["file"]: name for name, c in notes["clips"].items()}
    vocab = ClassVocabulary(name="site-zero-synthetic", version="1", labels=("Person",))

    def factory(camera: Any) -> Any:
        frames = truth[by_file[camera.video.name]]
        return OracleDetector(
            GroundTruthBoxes(
                boxes={int(f): np.array(b, np.float64) for f, b in frames.items()},
                labels={int(f): ["Person"] * len(b) for f, b in frames.items()},
                source_sha="site-zero-synthetic",
            ),
            vocab,
        )

    return calibration, factory, "Person", known


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data", type=Path, default=REAL_DATA)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "stand_in"
            calibration, factory, label, known = synthetic_inputs(data_dir)
            report = run_site_zero(
                data_dir,
                calibration,
                args.out,
                factory,
                label,
                cache_root=Path(tmp) / "cache",
                allow_dirty=True,
            )
            report["known_answers"] = known
    else:
        if args.calibration is None:
            p.error("--calibration is required for the real capture")
        from src.main import pretrained_detector_factory

        report = run_site_zero(
            args.data,
            args.calibration,
            args.out,
            pretrained_detector_factory(),
            "person",
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "site_zero_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
