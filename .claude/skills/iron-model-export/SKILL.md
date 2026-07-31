---
name: iron-model-export
description: Model export, conversion, and INT8 quantization checklist for project-iron (V-JEPA2, Depth-Anything-V2, CoTracker3, pose models → OpenVINO). MUST be consulted for ANY task involving exporting a model, converting to OpenVINO/ONNX, quantizing, changing input resolution or clip length, writing or editing a model wrapper, or debugging "the model runs but results look wrong". Triggers on: "export", "convert", "quantize", "INT8", "OpenVINO", "onnx", "xml model", "add a model", "wrapper", "change resolution", "change clip length". Silent export bugs cost 30-50% quality and never throw — this checklist is how they die.
---

# Iron Model Export — The Silent-Bug Killer

An exported vision transformer that is wrong still runs, still outputs plausible tensors, and
still passes smoke tests. Every item below is a real bug class that produces exactly that.

## Export Checklist (all items, every export, no exceptions)

1. **Position embedding interpolation.** Changing spatial resolution (e.g. 256→224) or clip
   length changes the token grid; pos-embeds MUST be interpolated at export. Verify by count:
   V-JEPA2 ViT-L = 16×16 spatial patches, tubelet 2 → at 224² input, 14×14=196 spatial ×
   (T/2) temporal tokens. Print expected vs actual token count in the export script; abort on
   mismatch.
2. **Which weights?** V-JEPA trains a context encoder and an EMA target encoder. Confirm and
   record in the export manifest which was exported. Wrong pick = total silent quality loss.
3. **PreprocessSpec travels with the model.** Serialize `{frames, stride, resolution, mean,
   std, channel_order, resize_policy, tubelet, patch_size}` next to the exported file. The
   wrapper LOADS this spec and asserts runtime config matches; hardcoding these in Python is
   forbidden — that's how `CLIP_FRAMES=4` shipped.
4. **Golden-vector test.** Fixed input → embedding must match a stored reference (produced by
   the reference PyTorch implementation) within tolerance. Commit the reference vector. This
   is the single test that catches items 1–3 when they slip.
5. **Clip semantics.** Video models: 16 frames at a stride spanning ~2 s of real time. 4
   consecutive frames is 0.13 s — a static image at video-model prices. Resolution matches the
   export; mismatch is a load-time error, not a warning.

## INT8 Quantization Gates (quantize LAST, after model choice is frozen)

1. **Calibration data = ≥512 real domain frames**, stratified across golden-set categories
   (night, rain, blur included). Random noise or generic web images as calibration data costs
   more accuracy than every other bug combined. Record calibration-set sha in the export
   manifest.
2. **Accuracy gate, in CI**: FP32↔INT8 per-patch cosine over ≥500 real frames — gate on the
   **1st percentile**, not the mean; AND task-level gate (retrieval mAP degradation < 1%, or
   depth AbsRel delta < declared bound). Both must pass or the export is rejected.
3. **Determinism note**: INT8 + multithreaded OpenVINO is deterministic only at fixed thread
   count. Record thread settings; the golden-vector test runs at the pinned count.
4. Re-quantization invalidates downstream indexes — see iron-provenance coupling law; the
   rebuild plan is part of the quantization PR.

## Wrapper Rules

Wrappers return contract types (`DepthField`, `PatchTokens` — see iron-contracts), apply the
loaded PreprocessSpec internally, expose `model_sha` and `preprocess_sha` properties, guard
outputs with finite checks (one NaN vector corrupts an entire FAISS index), and never contain
a hardcoded mean/std/resolution/frame-count. `ENABLE_MMAP` stays on for CPU; measure mmap
latency at **p99**, not mean — page faults on slow disks live in the tail.
