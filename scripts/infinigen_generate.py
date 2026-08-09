"""Generate one indoor Infinigen sample for the depth validity gate.

Runs in the isolated Infinigen venv (Python 3.11, ``bpy==4.2.0``), not the
pinned measurement env (Python 3.10). The pinned env stays untouched: Infinigen
pins ``numpy<2`` and pulls in Blender's Python, and mixing that with the eval
stack is how a benchmark ends up scoring the wrong library.

As of Day 18 the Infinigen venv lives outside this repo tree entirely, as a
sibling directory next to the repo root — it is a separate toolchain (Python
3.11 against this project's 3.10 pin), not a drift risk, and env_gate.py's
stray-venv scan should never see it. Create it yourself if it does not exist
on your machine: ``python3.11 -m venv ../project-iron-infinigen-venv`` from
the repo root, then install Infinigen's own requirements into it.

    ../project-iron-infinigen-venv/bin/python scripts/infinigen_generate.py \\
        --output data/infinigen_probe/scene_0001 --frames 8 --resolution 320 240

Output layout, consumed by ``scripts/run_infinigen_gate.py``:

    <output>/
        frame_0001.npz  (rgb: [H,W,3] uint8, depth_metres: [H,W] float32)
        frame_0002.npz
        ...
        provenance.json (Infinigen version, seed, resolution, samples, elapsed)

Depth is written in metres after inversion from Infinigen's z-buffer, so the
gate never touches an .exr and no OpenEXR reader has to be added to the pinned
stack.

Honesty note on scope
---------------------
Infinigen-Indoors 1.15 procedurally generates rooms, furniture, materials and
lighting. It does NOT generate humans in the default indoor pipeline — the
human-body assets in Infinigen live in a separate branch and are not
commercially cleared. This script does not place people, and the run report
must therefore say plainly: **human-appearance capabilities remain untested by
this generation**. Depth and non-human appearance are what this sample can
speak to.

Exit codes:
    0  frames written
    1  Infinigen invocation failed — the failure is the finding
    2  bpy/infinigen not importable in this venv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def check_environment() -> tuple[bool, str]:
    """Confirm bpy and infinigen import in the current interpreter."""
    try:
        import bpy  # noqa: F401
    except Exception as exc:
        return False, f"bpy not importable: {exc}"
    try:
        import infinigen  # noqa: F401
    except Exception as exc:
        return False, f"infinigen not importable: {exc}"
    return True, "ok"


def render_scene(
    output: Path, seed: int, frames: int, width: int, height: int, samples: int
) -> subprocess.CompletedProcess:
    """Invoke Infinigen's indoors example generator as a subprocess.

    Kept as a subprocess for two reasons: a Blender crash or segfault (macOS
    CPU renders take these more than they should) surfaces as a non-zero exit
    code instead of dropping the whole harness; and the child needs a
    different working directory than the caller.

    Infinigen 1.12 registers its ``.gin`` config folders as relative paths
    (``infinigen_examples/configs_indoor``, ``infinigen_examples/configs_nature``)
    and ``gin.parse_config_file`` resolves ``include`` statements against those
    folders. From any other CWD the include of ``surface_registry.gin`` raises
    ``TypeError: expected str, bytes or os.PathLike object, not NoneType`` deep
    in ``gin.resource_reader`` — the include never resolves. Running from the
    site-packages root fixes this without patching upstream.
    """
    output.mkdir(parents=True, exist_ok=True)
    output_abs = output.resolve() / "_infinigen"
    import infinigen_examples

    site_packages = Path(infinigen_examples.__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "infinigen_examples.generate_indoors",
        "--output_folder",
        str(output_abs),
        "--seed",
        str(seed),
        "--task",
        "coarse",
        "populate",
        "fine_terrain",
        "render",
        "-g",
        "singleroom.gin",
        "fast_solve.gin",
        "no_objects.gin",
        "--overrides",
        f"execute_tasks.frame_range=[1,{frames}]",
        f"execute_tasks.generate_resolution=[{width},{height}]",
        f"configure_render_cycles.num_samples={samples}",
        f"configure_render_cycles.min_samples={samples}",
        f"configure_render_cycles.adaptive_threshold=0.1",
        f"compose_indoors.terrain_enabled=False",
    ]
    print("running (cwd=", site_packages, "):", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=site_packages)


def pack_frames(source: Path, output: Path) -> int:
    """Read Infinigen's .exr depth + .png rgb, write one .npz per frame.

    Returns the number of frames written. Does the conversion in this venv
    because it has OpenEXR (a dep of Infinigen) and the pinned venv does not.
    """
    import numpy as np
    import OpenEXR
    import Imath
    from PIL import Image

    frames_dir = source / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Infinigen produced no frames/ under {source}")

    depth_files = sorted(frames_dir.glob("Depth_*.exr"))
    written = 0
    for depth_file in depth_files:
        name = depth_file.stem  # Depth_camera_0_frame_0001
        _, cam, _, frame = name.split("_", 3)
        image_file = frames_dir / f"Image_{cam}_frame_{frame}.png"
        if not image_file.exists():
            continue

        exr = OpenEXR.InputFile(str(depth_file))
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(
            height, width
        )

        rgb = np.asarray(Image.open(image_file).convert("RGB"), dtype=np.uint8)

        np.savez_compressed(
            output / f"frame_{frame}.npz",
            rgb=rgb,
            depth_metres=depth.astype(np.float32),
        )
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=[320, 240])
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help=(
            "Cycles render samples. Kept low deliberately: this sample exists "
            "to run through the validity gate, not to be photographed. Even so, "
            "CPU render time is measured in minutes per frame at 320x240."
        ),
    )
    args = parser.parse_args(argv)

    ok, why = check_environment()
    if not ok:
        print(f"environment check failed: {why}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    result = render_scene(
        args.output,
        args.seed,
        args.frames,
        args.resolution[0],
        args.resolution[1],
        args.samples,
    )
    elapsed = time.perf_counter() - started
    stdout_tail = "\n".join(result.stdout.splitlines()[-40:])
    stderr_tail = "\n".join(result.stderr.splitlines()[-40:])
    if result.returncode != 0:
        (args.output / "generate_stdout.log").write_text(result.stdout)
        (args.output / "generate_stderr.log").write_text(result.stderr)
        print(
            f"infinigen render failed (exit {result.returncode}, "
            f"{elapsed:.0f}s). Tail of stderr:\n{stderr_tail}",
            file=sys.stderr,
        )
        return 1

    try:
        written = pack_frames(args.output / "_infinigen", args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"pack_frames failed: {exc}", file=sys.stderr)
        return 1

    from importlib.metadata import version

    provenance = {
        "infinigen_version": version("infinigen"),
        "bpy_version": version("bpy"),
        "seed": args.seed,
        "requested_frames": args.frames,
        "written_frames": written,
        "resolution": args.resolution,
        "cycles_samples": args.samples,
        "elapsed_seconds": round(elapsed, 2),
        "generator_cmd_tail": stdout_tail[-500:],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"wrote {written} frames in {elapsed:.0f}s to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
