"""Isolated Cycles raytracing throughput probe — NOT an Infinigen benchmark.

Renders a trivial scene (one cube, one sun light) at a chosen resolution and
sample count to measure per-frame raytracing cost on whatever Cycles device
is configured, separate from scene composition cost. Written for the Day-12
Infinigen throughput question (docs/day12/infinigen_throughput.md): Day 11's
21m35s session bought partial constraint-solving progress and zero rendered
pixels, so the render stage's own cost had never been measured. This probe
measures it in isolation, cheaply, without running Infinigen at all.

Must run inside a Blender-embedded Python (``bpy`` is not installable via
pip in general — the project's Infinigen environment provides it). As of
Day 18 that environment lives outside this repo tree, as a sibling directory
next to the repo root:

    ../project-iron-infinigen-venv/bin/python scripts/probe_cycles_throughput.py

The FIRST render at any sample count includes one-time Cycles kernel
compilation / Metal shader compilation, which dwarfs every subsequent
render (a documented Cycles behaviour — see the "Loading denoising kernels"
log line). This probe renders 3 frames per sample count precisely so that
first-render outlier is visible and excludable rather than silently
averaged into a fake per-frame cost.

Caveat, repeated because it matters: a bare cube is not a furnished room.
Cycles cost scales with polygon count, material complexity and light-bounce
depth — a real Infinigen interior has far more of all three. Treat this
probe's numbers as a floor on render-stage cost, not a prediction of it.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--samples", type=int, nargs="+", default=[4, 64],
        help="Cycles sample counts to probe, each timed across 3 renders.",
    )
    parser.add_argument("--device-type", default="METAL", choices=["METAL", "CUDA", "OPTIX", "CPU"])
    parser.add_argument("--out-prefix", default="/tmp/cycles_probe_frame_")
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    bpy.ops.object.light_add(type="SUN")
    bpy.ops.object.camera_add(location=(4, -4, 3), rotation=(1.1, 0, 0.8))
    camera_obj = next(o for o in bpy.context.scene.objects if o.type == "CAMERA")
    bpy.context.scene.camera = camera_obj

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if args.device_type != "CPU":
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = args.device_type
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
    else:
        scene.cycles.device = "CPU"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.filepath = args.out_prefix

    import time

    print(f"device_type={args.device_type} resolution={args.width}x{args.height}")
    for samples in args.samples:
        scene.cycles.samples = samples
        times = []
        for _ in range(3):
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            times.append(time.time() - t0)
        steady_state = times[1:]  # exclude the first-render kernel-compile outlier
        mean_steady = sum(steady_state) / len(steady_state) if steady_state else float("nan")
        print(
            f"samples={samples}  per-frame seconds={[round(t, 3) for t in times]}  "
            f"steady_state_mean={mean_steady:.3f} (first render excluded — "
            "kernel/shader compilation, not raytracing)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
