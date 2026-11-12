*Lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration given; not evidence about real cameras.*

Event matching tolerance: 2.0 s (`src.pipeline.evaluate.EVENT_TOLERANCE_S`), same verb, zone and attributed GT subject, one-to-one greedy by time gap.

## Warehouse_022, pretrained, 4 cameras x 1800 frames

Source: `artifacts/day44/frontier_w022_pretrained_iou.json`

| candidate | invocations (per cam-min) | precision | recall | vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | ID sw IoU-only [95%] | RMSE [0,10) est / raw (n) | RMSE [10,20) est / raw (n) | RMSE [20,30) est / raw (n) | RMSE [30,inf) est / raw (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 1154 (288) | 56/66 | 56/78 | +8 / -10, -0.026 [-0.118, +0.078] | 40 [24, 57] | 40 [24, 57] | 0.474 / 0.478 (327) | 0.238 / 0.227 (948) | 0.511 / 0.416 (808) | 0.753 / 0.571 (675) |
| gate_default | 4172 (1043) | 36/44 | 36/78 | +1 / -23, -0.282 [-0.359, -0.182] | 84 [47, 124] | 84 [47, 124] | 0.510 / 0.512 (1781) | 0.252 / 0.235 (4096) | 0.473 / 0.433 (2619) | 0.604 / 0.519 (2703) |
| gate_keepalive | 6011 (1503) | 53/61 | 53/78 | +0 / -5, -0.064 [-0.129, +0.000] | 88 [43, 140] | 88 [43, 140] | 0.493 / 0.494 (1987) | 0.245 / 0.228 (5403) | 0.518 / 0.406 (4375) | 0.691 / 0.552 (3576) |
| gate_off | 7200 (1800) | 58/66 | 58/78 | reference | 93 [48, 144] | 93 [48, 144] | 0.493 / 0.494 (1987) | 0.246 / 0.231 (5697) | 0.506 / 0.394 (4859) | 0.679 / 0.529 (4064) |
| motion_adaptive | 705 (176) | 32/42 | 32/78 | +4 / -30, -0.333 [-0.434, -0.203] | 50 [30, 72] | 50 [30, 72] | 0.508 / 0.512 (294) | 0.245 / 0.234 (691) | 0.429 / 0.400 (445) | 0.659 / 0.528 (458) |
| stride_15 | 480 (120) | 40/53 | 40/78 | +4 / -22, -0.231 [-0.329, -0.148] | 102 [63, 145] | 102 [63, 145] | 0.540 / 0.544 (100) | 0.235 / 0.237 (350) | 0.411 / 0.443 (311) | 0.637 / 0.526 (242) |
| stride_6 | 1200 (300) | 55/63 | 55/78 | +7 / -10, -0.038 [-0.110, +0.055] | 44 [25, 65] | 44 [25, 65] | 0.467 / 0.471 (323) | 0.236 / 0.229 (947) | 0.500 / 0.380 (806) | 0.705 / 0.541 (685) |

Stride 15 vs gate-off, GT events gate-off matched and stride 15 did not: lost 14 (4 fired within tolerance on a track attributed to another GT subject or none, 10 with no such event at all); 8 with the same person, zone and verb only more than 2.0 s away (gaps s: 2.30, 3.93, 5.10, 16.70, 17.17, 17.17, 27.53, 36.87; a different visit is the likelier reading than a late event); taken by another GT event 0. Matched events: signed time shift p50 0.383 s, |shift| p90 1.673 s.
