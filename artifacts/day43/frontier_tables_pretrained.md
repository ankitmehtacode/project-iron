*Lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration given; not evidence about real cameras.*

Event matching tolerance: 2.0 s (`src.pipeline.evaluate.EVENT_TOLERANCE_S`), same verb, zone and attributed GT subject, one-to-one greedy by time gap.

## Warehouse_022, pretrained, 4 cameras x 1800 frames

Source: `artifacts/day43/frontier_w022_pretrained.json`

| candidate | invocations (per cam-min) | precision | recall | vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | ID sw IoU-only [95%] | RMSE [0,10) est / raw (n) | RMSE [10,20) est / raw (n) | RMSE [20,30) est / raw (n) | RMSE [30,inf) est / raw (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 1156 (289) | 59/69 | 59/78 | +1 / -7, -0.077 [-0.159, +0.000] | 83 [41, 136] | 40 [21, 63] | 0.495 / 0.500 (330) | 0.238 / 0.231 (959) | 0.482 / 0.455 (832) | 0.766 / 0.652 (773) |
| gate_default | 4172 (1043) | 42/53 | 42/78 | +0 / -23, -0.295 [-0.397, -0.194] | 259 [124, 429] | 84 [47, 129] | 0.557 / 0.559 (1823) | 0.251 / 0.234 (4149) | 0.463 / 0.465 (2759) | 1.048 / 0.691 (3230) |
| gate_keepalive | 6022 (1506) | 59/81 | 59/78 | +0 / -6, -0.077 [-0.154, -0.013] | 399 [195, 656] | 88 [41, 150] | 0.537 / 0.539 (2029) | 0.245 / 0.228 (5459) | 0.520 / 0.460 (4540) | 1.207 / 0.753 (4251) |
| gate_off | 7200 (1800) | 65/87 | 65/78 | reference | 453 [233, 721] | 93 [46, 154] | 0.537 / 0.539 (2029) | 0.246 / 0.231 (5757) | 0.508 / 0.444 (5024) | 1.147 / 0.718 (4820) |
| motion_adaptive | 705 (176) | 35/45 | 35/78 | +1 / -31, -0.385 [-0.507, -0.273] | 87 [45, 139] | 50 [28, 77] | 0.534 / 0.538 (299) | 0.244 / 0.234 (698) | 0.455 / 0.401 (461) | 0.710 / 0.592 (518) |
| stride_15 | 480 (120) | 46/59 | 46/78 | +2 / -21, -0.244 [-0.381, -0.121] | 93 [62, 128] | 102 [68, 140] | 0.569 / 0.574 (102) | 0.237 / 0.239 (349) | 0.467 / 0.417 (320) | 0.687 / 0.557 (264) |
| stride_6 | 1200 (300) | 60/72 | 60/78 | +2 / -7, -0.064 [-0.157, +0.012] | 94 [44, 156] | 44 [22, 72] | 0.495 / 0.499 (331) | 0.236 / 0.232 (959) | 0.473 / 0.413 (829) | 0.693 / 0.621 (771) |

Stride 15 vs gate-off, GT events gate-off matched and stride 15 did not: lost 14 (6 fired within tolerance on a track attributed to another GT subject or none, 8 with no such event at all); 7 with the same person, zone and verb only more than 2.0 s away (gaps s: 2.30, 3.93, 11.50, 11.50, 17.20, 23.43, 31.03; a different visit is the likelier reading than a late event); taken by another GT event 0. Matched events: signed time shift p50 0.500 s, |shift| p90 1.000 s.
