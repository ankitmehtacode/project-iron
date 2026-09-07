# Weight status — canonical, citable, re-verifiable

Day 34, Objective 1. This has been asserted wrong twice in two different
directions across recent sessions — Day 33 assumed CoTracker3 had executed
for real (it hadn't; every `requires_weights` test skipped on the missing
VJEPA IR), and the correction risked over-rotating into "VJEPA is just
generally absent," collapsing three separate VJEPA-related artifacts that
are NOT all in the same state. This document exists so the status stops
being re-derived from memory each time it comes up: run the command block
below rather than trust the prose.

## The three separate questions, because they are not one fact

1. **Do the raw V-JEPA2 model weights exist** (Day-3's scripted
   `HuggingFace` download)? **Yes.**
2. **Does an FP32 OpenVINO export of them exist** (Day 3/8's
   `scripts/export_vjepa_ov.py`)? **Yes**, dated 2026-07-31, verified
   against a PyTorch reference (`export_manifest.json`).
3. **Does the PRODUCTION INT8-quantized artifact
   (`models/int8/vjepa2_vitl_int8.xml`/`.bin`) exist?** **No** — and per
   ADR 0008 this is permanently unattributable, not merely un-exported: the
   artifact production actually ran inference against was never captured
   with a matching manifest, and re-exporting a fresh INT8 IR today would
   be a *different* artifact standing in for the lost one, not a recovery
   of it. This is the row `scripts/env_gate.py` gates the forensic verdict
   on (`blocks="objective-1"`) and the only one of the three that matters
   for that verdict.

**CoTracker3's checkpoint is present and loads.** Unrelated to any of the
three VJEPA questions above — conflating "a VJEPA artifact is missing" with
"no weights exist at all" is exactly the kind of collapse that produced
Day 33's wrong-direction correction.

## The check, embedded — run this, don't read the numbers below as fact

```bash
source .venv-pinned/bin/activate

# Structural gate (this is the check env_gate.py already runs as Row 2 --
# see "Structural enforcement" below; this doc does not duplicate it, it
# points at it):
python scripts/env_gate.py --json | python3 -c "
import json, sys
gate = json.load(sys.stdin)
for row in gate['rows']:
    if row['name'].startswith('model'):
        print(row['name'], '->', 'PASS' if row['passed'] else 'FAIL', '--', row['detail'])
"

# Raw weights + FP32 export, direct:
python3 -c "
from pathlib import Path
for p in [
    'models/weights/vjepa2_vitl/model.safetensors',
    'models/export/2026-07-31/vjepa2_vitl_fp32.xml',
    'models/export/2026-07-31/vjepa2_vitl_fp32.bin',
    'models/int8/vjepa2_vitl_int8.xml',
    'models/int8/vjepa2_vitl_int8.bin',
    'models/weights/cotracker3/scaled_offline.pth',
]:
    path = Path(p)
    print(p, '->', 'EXISTS' if path.exists() else 'ABSENT',
          f'{path.stat().st_size} bytes' if path.exists() else '')
"

# CoTracker3 checkpoint actually loads (not just present on disk):
python3 -c "
import torch
sd = torch.load('models/weights/cotracker3/scaled_offline.pth', map_location='cpu', weights_only=False)
print('loads OK:', type(sd).__name__, 'with', len(sd), 'keys')
"
```

## Result, as of this check (2026-09-07, this session)

| artifact | status | path | size | sha256 |
| --- | --- | --- | ---: | --- |
| VJEPA2 raw weights (`model.safetensors`) | EXISTS | `models/weights/vjepa2_vitl/model.safetensors` | 1,303,947,864 B | `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b` (`export_manifest.json`'s own record, not re-hashed here — see note below) |
| VJEPA2 FP32 OpenVINO export (`.xml`) | EXISTS | `models/export/2026-07-31/vjepa2_vitl_fp32.xml` | 4,541,054 B | `5ca2d3da552016a67390be14b6dc3fd9dc1dbd47c7b78b24fcd9a96b0393a155` |
| VJEPA2 FP32 OpenVINO export (`.bin`) | EXISTS | `models/export/2026-07-31/vjepa2_vitl_fp32.bin` | 1,215,541,688 B | `2130cd5fb9674bcd64fffcaffd789d11900a285f8d9206b646b947091e6fdf3b` |
| **VJEPA2 PRODUCTION INT8 IR (`.xml`/`.bin`)** | **ABSENT** | `models/int8/vjepa2_vitl_int8.{xml,bin}` | — | — (ADR 0008: permanently unattributable) |
| CoTracker3 checkpoint (`scaled_offline.pth`) | EXISTS, loads | `models/weights/cotracker3/scaled_offline.pth` | 101,890,938 B | `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` (freshly computed this session — `models/weights/hashes.txt` exists but is currently empty, not a source for this row) |

The FP32 export's own hashes are quoted from `models/export/2026-07-31/
export_manifest.json` rather than re-computed here, since that manifest is
itself the citable record for that artifact (`scripts/export_vjepa_ov.py`
wrote it, with a PyTorch-reference verification block: max abs deviation
2.99e-4, cosine p1 0.99999999907). CoTracker3's hash was not previously
recorded anywhere citable (`models/weights/hashes.txt` is empty) and was
computed fresh above.

## Structural enforcement

`scripts/env_gate.py`'s Row 2 (`check_models`) already reports all three
model-presence facts relevant to a gated run — `PRODUCTION vjepa_xml`,
`PRODUCTION vjepa_bin`, `cotracker_checkpoint` — with size and a truncated
sha256, on every invocation. **This was already structural before today**;
Objective 1 found no gap to close there, only a canonical prose record to
write so this stops being re-derived from memory. Nothing in `env_gate.py`
needed to change.

What is NOT gated anywhere: the raw-weights and FP32-export rows above
(questions 1 and 2). They are not part of `env_gate.py`'s check because
nothing downstream depends on them directly — only the PRODUCTION INT8
artifact (question 3) and the CoTracker3 checkpoint gate anything. Adding
gate rows for the FP32 export would be gating a fact nothing consumes,
which is not what Row 2 exists for; if that changes, add the row then.
