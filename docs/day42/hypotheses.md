# Day 42 — hypotheses H1–H10 (Objective B1)

Each assertion in the Day-42 prompt was a hypothesis until checked here. Nothing
below was copied into a registry entry before its row was settled.

| | hypothesis | verdict | how it was settled |
|---|---|---|---|
| H1 | no working detector exists; `DetectorStage` raises `NotImplementedError` | **VERIFIED** | `src/cascade/stages.py:49` raises it; `grep -rlniE "yolo\|detr\|ultralytics\|fasterrcnn\|ssdlite\|detector" src scripts` finds only docstrings, the gate and the stub |
| H2 | x86_64; torch 2.2.0 CPU build; installed transformers imports a real-time DETR-family detector | **VERIFIED, restated** | `uname -m` → `x86_64` (i7-8750H). `torch.version.cuda` → `None`. Random-init forward pass at 640x640: in the **lockfile** environment (transformers 4.50.3) `RTDetrForObjectDetection` and `RTDetrV2ForObjectDetection` run; `DFineForObjectDetection` does not exist (`no attribute DFineConfig`). It ran only in the drifted `.venv-pinned` (transformers 4.57.6, tokenizers 0.22.2). D-FINE is therefore off the shortlist: adopting it would mean upgrading the pinned tokenizers |
| H3 | `pycocotools` installs at an exact pin and agrees with a hand-computed AP | **VERIFIED** | `pip install pycocotools==2.0.11` (binary wheel, no new transitive dependency); `tests/test_coco_metrics.py`: AP, AP50 = 56/101 and AR@100 = 2/3, each to 1e-9 |
| H4 | candidate licences: RT-DETR family, YOLOX, D-FINE Apache-2.0; Ultralytics AGPL-3.0 | **UNVERIFIABLE HERE (human signature required)** | metadata only, recorded as `hypothesis_class`: Hub card `license: apache-2.0` for `PekingU/rtdetr_v2_r18vd@5650961749fa93567c0d46fc7f43ea4f9e914107` and `ustc-community/dfine-small-coco@f79e65b5…`; GitHub licence API `Apache-2.0` for YOLOX `LICENSE@6ddff4824372906469a7fae2dc3206c7aa4bbaee`. Ultralytics not read; excluded by rule and registered `ship_blocked: true` |
| H5 | detector boxes are in model-input pixel space (stretch or letterbox), so a typed `AffineTransform` back to 1920x1080 is required | **VERIFIED for RT-DETR(v2); YOLOX pending its code read** | `RTDetrImageProcessor()` defaults: `do_resize=True`, `size=640x640`, `do_pad=False` (stretch), `rescale_factor=1/255`, `do_normalize=False`; `post_process_object_detection` scales normalised cxcywh by `target_sizes`, which is correct only for a stretch. Each signed model's own `preprocessor_config.json` is re-read at load time (B3) |
| H6 | CHIRLA tracking configs hold full frames with person boxes | **FALSIFIED** | every row of all six configs (17,437 rows; parquet `resolution` column, images not decoded) is an identity crop; the per-camera annotation JSONs the rows reference were not fetched. No CHIRLA detection check today |
| H7 | SmartSpaces GT classes: Person plus forklift, NovaCarter, Transporter, FourierGR1T2, AgilityDigit | **VERIFIED, one addition** | `class_names` of all 23 cached GT files: Person 23 scenes, Forklift 17, **PalletTruck 13**, Transporter 10, AgilityDigit 9, FourierGR1T2 9, NovaCarter 8 |
| H8 | a COCO-trained detector has no forklift or AMR class | **VERIFIED** | `config.json` of `PekingU/rtdetr_v2_r18vd` at the pinned commit (metadata, not weights): 80 classes; nearest strings are `truck` (road vehicle) and `fork` (cutlery) |
| H9 | Day-41 contact sheets are excluded by a versioned `.gitignore` rule | **FALSIFIED** | `git check-ignore -v` → `.git/info/exclude:14`. Fixed in `2ad2c57` |
| H10 | MOG2 history / learning rate readable from the gate code | **VERIFIED** | `src/cascade/motion.py:268` `createBackgroundSubtractorMOG2(history=250, varThreshold=16, detectShadows=True)`; `apply(gray)` at :277 passes no learning rate, so OpenCV's automatic `1/min(frames_seen, 250)` applies |

## Shortlist carried to the B2 human checkpoint

At most three, permissive by hypothesis and runnable on the pinned stack:

1. **RT-DETRv2-S (COCO)**: `PekingU/rtdetr_v2_r18vd` @ `5650961749fa93567c0d46fc7f43ea4f9e914107`, transformers 4.50.3.
2. **YOLOX-S (COCO)**: release `0.1.1rc0` asset `yolox_s.onnx`, run through the pinned openvino 2024.6.0 (PyTorch-free). The release publishes no checksum, so the first fetch is trust-on-first-use.

Not shortlisted: D-FINE (not runnable on the pinned stack, see H2) and Ultralytics YOLO (AGPL, by rule).
