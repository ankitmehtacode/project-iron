---
name: iron-contracts
description: Typed boundary contracts for project-iron — units, coordinate frames, intrinsics, transforms, patch-token shapes. MUST be consulted before writing or modifying ANY function that passes arrays between modules, touches depth values, pixel coordinates, 3D points, camera intrinsics, resizing, cropping, patch embeddings, or unprojection. Also triggers on: "add a model wrapper", "fuse tracks with depth", "map embeddings to points", "resize", "projector", "coordinate", "meters", "disparity". Skipping this skill is how the off-by-2 and units bugs happened.
---

# Iron Contracts — Make Illegal States Unrepresentable

Every production bug found in this repo's audit was a *boundary* bug: wrong units crossing a
function call, wrong pixel frame crossing a resize, wrong temporal token count crossing a
docstring. The fix is structural, not vigilance. These rules are law.

## The Prime Rule

**No raw `np.ndarray` or `torch.Tensor` crosses a public module boundary.** Arrays travel inside
typed envelopes from `src/contracts/` that carry the metadata needed to interpret them:

| Envelope | Carries | Kills which bug |
|---|---|---|
| `DepthField` | data, `units`, `geometry`, `valid_mask` | disparity labeled "meters" |
| `PatchTokens` | data, `TemporalSpan(tubelet)`, `grid`, `geometry`, `encoder_sha` | temporal off-by-2 |
| `Intrinsics` | fx, fy, cx, cy, distortion, `valid_for: FrameGeometry` | stale fx after resize |
| `AffineTransform` | stage-space → canonical-space mapping | letterbox/stretch drift |

Consumers **raise, never coerce**: `unproject()` raises `UnitsError` on non-metric depth. Do not
add a convenience auto-conversion — silent coercion recreates the bug with extra steps.

## Coordinate Frame Law

1. **One canonical frame**: the original video's pixel space, pixel centers at half-integer
   coordinates. Every stage output carries an `AffineTransform` back to it. No implicit resizes,
   ever — if you resize, you construct and attach the transform in the same statement.
2. Intrinsics are geometry-specific. Resizing a frame without `intrinsics.rescaled_to(new_geom)`
   is a P0 defect. Grep check before commit: any `cv2.resize` / `F.interpolate` near camera code
   must be adjacent to a transform/intrinsics update.
3. Undistortion happens at ingest, once, recorded in provenance. Distortion is not affine and
   must never enter the transform algebra.
4. World frame (multi-camera): `CameraToWorld` rigid transforms only, per `site_id`, versioned
   with `twin_rev`. Timeline key is `(site_id, ts_ns)` — **frame indices are presentation-only**;
   a `frame_idx` join anywhere outside the UI layer is a rejected PR.

## Temporal Law

`PatchTokens.__post_init__` asserts `n_temporal == frames_covered // tubelet`. If your change
makes this assertion inconvenient, your change has the bug, not the assertion. V-JEPA2 tubelet
is 2; 4 frames → 2 temporal tokens, never 4.

## Sampling Law

- Patch features at track points: **bilinear interpolation** on the patch grid at subpixel
  location — never `floor(x/patch)`. L2-normalize before any index insert.
- Never average features or depth across a depth discontinuity (check `depth_edge` mask).
- Clamp tracker outputs to frame bounds and flag; out-of-bounds coordinates exist and will
  otherwise wrap into a wrong patch index silently.

## When Adding a New Contract Type

Frozen dataclass, `__post_init__` shape/consistency validation, full type hints, and a
Hypothesis property test in `tests/test_contracts_properties.py` (round-trip, inverse,
composition — whatever invariants apply). A contract without a property test is a comment,
not a contract.
