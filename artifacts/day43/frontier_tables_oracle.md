*Lane S synthetic (NVIDIA PhysicalAI-SmartSpaces), calibration given; not evidence about real cameras.*

Event matching tolerance: 2.0 s (`src.pipeline.evaluate.EVENT_TOLERANCE_S`), same verb, zone and attributed GT subject, one-to-one greedy by time gap.

## Warehouse_022, oracle, 4 cameras x 1800 frames

Source: `artifacts/day43/frontier_w022_oracle.json`

| candidate | invocations (per cam-min) | precision | recall | vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | ID sw IoU-only [95%] | RMSE [0,10) est / raw (n) | RMSE [10,20) est / raw (n) | RMSE [20,30) est / raw (n) | RMSE [30,inf) est / raw (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 1177 (294) | 72/77 | 72/78 | +0 / -4, -0.051 [-0.104, -0.012] | 29 [20, 38] | 21 [12, 32] | 0.573 / 0.576 (343) | 0.561 / 0.554 (1083) | 0.869 / 0.811 (930) | 0.591 / 0.506 (979) |
| gate_default | 4172 (1043) | 47/55 | 47/78 | +0 / -29, -0.372 [-0.482, -0.254] | 36 [24, 49] | 36 [23, 51] | 0.661 / 0.660 (1909) | 0.588 / 0.570 (4714) | 0.495 / 0.348 (3117) | 0.588 / 0.497 (4050) |
| gate_keepalive | 6061 (1515) | 70/73 | 70/78 | +0 / -6, -0.077 [-0.154, -0.013] | 27 [15, 41] | 25 [14, 39] | 0.652 / 0.651 (2130) | 0.632 / 0.616 (6225) | 0.991 / 0.918 (5133) | 0.690 / 0.550 (5292) |
| gate_off | 7200 (1800) | 76/79 | 76/78 | reference | 34 [20, 51] | 32 [18, 48] | 0.653 / 0.652 (2147) | 0.618 / 0.603 (6628) | 0.951 / 0.878 (5635) | 0.676 / 0.529 (5906) |
| motion_adaptive | 706 (176) | 43/49 | 43/78 | +0 / -33, -0.423 [-0.538, -0.299] | 33 [22, 45] | 41 [25, 59] | 0.597 / 0.600 (310) | 0.503 / 0.492 (775) | 0.406 / 0.275 (526) | 0.535 / 0.452 (675) |
| stride_15 | 480 (120) | 48/58 | 48/78 | +0 / -28, -0.359 [-0.457, -0.276] | 123 [83, 166] | 138 [93, 186] | 0.584 / 0.590 (102) | 0.577 / 0.576 (382) | 0.888 / 0.789 (355) | 0.377 / 0.301 (346) |
| stride_6 | 1200 (300) | 72/77 | 72/78 | +0 / -4, -0.051 [-0.104, -0.012] | 29 [20, 39] | 20 [9, 32] | 0.573 / 0.576 (343) | 0.558 / 0.553 (1081) | 0.874 / 0.811 (928) | 0.597 / 0.506 (981) |

Stride 15 vs gate-off, GT events gate-off matched and stride 15 did not: lost 22 (6 fired within tolerance on a track attributed to another GT subject or none, 16 with no such event at all), shifted beyond 2.0 s 6 (gaps s: 4.07, 4.43, 5.03, 11.00, 17.17, 31.53), taken by another GT event 0. Matched events: signed time shift p50 0.283 s, |shift| p90 0.867 s.

## Warehouse_020, oracle, 4 cameras x 1800 frames

Source: `artifacts/day43/frontier_020_oracle.json`

| candidate | invocations (per cam-min) | precision | recall | vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | ID sw IoU-only [95%] | RMSE [0,10) est / raw (n) | RMSE [10,20) est / raw (n) | RMSE [20,30) est / raw (n) | RMSE [30,inf) est / raw (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 1200 (300) | 27/36 | 27/45 | +0 / -2, -0.044 [-0.109, +0.000] | 254 [196, 319] | 390 [295, 493] | 0.858 / 0.860 (779) | 2.961 / 2.996 (1707) | 2.756 / 3.200 (1132) | 20.392 / 20.382 (5414) |
| gate_default | 6544 (1636) | 28/42 | 28/45 | +0 / -1, -0.022 [-0.075, +0.000] | 278 [221, 335] | 278 [219, 337] | 0.914 / 0.922 (4628) | 4.083 / 4.121 (9922) | 2.960 / 2.866 (6318) | 21.607 / 21.640 (33529) |
| gate_keepalive | 6959 (1740) | 28/41 | 28/45 | +0 / -1, -0.022 [-0.075, +0.000] | 282 [222, 341] | 283 [221, 344] | 0.914 / 0.922 (4629) | 3.997 / 4.034 (10385) | 3.191 / 3.405 (6677) | 22.107 / 22.109 (34135) |
| gate_off | 7200 (1800) | 29/42 | 29/45 | reference | 282 [222, 341] | 283 [221, 344] | 0.904 / 0.911 (4798) | 3.976 / 4.013 (10495) | 3.171 / 3.384 (6947) | 22.036 / 22.037 (34681) |
| motion_adaptive | 1095 (274) | 27/37 | 27/45 | +0 / -2, -0.044 [-0.114, +0.000] | 249 [191, 314] | 337 [253, 426] | 0.871 / 0.872 (746) | 2.878 / 2.911 (1618) | 2.383 / 2.329 (1034) | 19.639 / 19.656 (5254) |
| stride_15 | 480 (120) | 9/16 | 9/45 | +0 / -20, -0.444 [-0.632, -0.268] | 609 [469, 752] | 765 [601, 935] | 0.716 / 0.716 (250) | 2.979 / 3.012 (565) | 2.783 / 3.190 (409) | 20.520 / 20.526 (1826) |
| stride_6 | 1200 (300) | 27/36 | 27/45 | +0 / -2, -0.044 [-0.109, +0.000] | 254 [196, 319] | 390 [295, 493] | 0.858 / 0.860 (779) | 2.961 / 2.996 (1707) | 2.756 / 3.200 (1132) | 20.392 / 20.382 (5414) |

Stride 15 vs gate-off, GT events gate-off matched and stride 15 did not: lost 14 (1 fired within tolerance on a track attributed to another GT subject or none, 13 with no such event at all), shifted beyond 2.0 s 6 (gaps s: 2.73, 3.50, 4.10, 5.43, 26.60, 46.27), taken by another GT event 0. Matched events: signed time shift p50 0.500 s, |shift| p90 1.547 s.

## Warehouse_000, oracle, 4 cameras x 1800 frames

Source: `artifacts/day43/frontier_000_oracle.json`

| candidate | invocations (per cam-min) | precision | recall | vs gate-off: +gained / -lost, Δrecall [95%] | ID sw pipeline [95%] | ID sw IoU-only [95%] | RMSE [0,10) est / raw (n) | RMSE [10,20) est / raw (n) | RMSE [20,30) est / raw (n) | RMSE [30,inf) est / raw (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 1200 (300) | 31/56 | 31/54 | +5 / -10, -0.093 [-0.200, +0.019] | 368 [299, 442] | 588 [471, 711] | 1.512 / 1.514 (363) | 0.438 / 0.627 (966) | 3.420 / 3.492 (1293) | 11.405 / 11.632 (8367) |
| gate_default | 5128 (1282) | 27/65 | 27/54 | +0 / -9, -0.167 [-0.281, -0.064] | 394 [317, 476] | 388 [308, 469] | 1.592 / 1.593 (2167) | 0.545 / 0.701 (5088) | 2.717 / 2.905 (6577) | 12.633 / 12.907 (46802) |
| gate_keepalive | 7020 (1755) | 35/75 | 35/54 | +0 / -1, -0.019 [-0.061, +0.000] | 437 [360, 518] | 426 [346, 510] | 1.614 / 1.615 (2294) | 0.520 / 0.680 (5710) | 2.534 / 2.715 (7725) | 12.830 / 13.063 (53223) |
| gate_off | 7200 (1800) | 36/76 | 36/54 | reference | 439 [361, 520] | 428 [347, 512] | 1.607 / 1.607 (2316) | 0.513 / 0.671 (5890) | 3.457 / 3.591 (7868) | 12.821 / 13.058 (53869) |
| motion_adaptive | 861 (215) | 24/47 | 24/54 | +0 / -12, -0.222 [-0.321, -0.132] | 333 [263, 411] | 536 [419, 660] | 1.523 / 1.525 (341) | 0.434 / 0.590 (836) | 2.636 / 2.814 (1084) | 11.186 / 11.374 (7267) |
| stride_15 | 480 (120) | 15/29 | 15/54 | +0 / -21, -0.389 [-0.483, -0.283] | 715 [589, 850] | 958 [796, 1134] | 1.151 / 1.151 (108) | 0.345 / 0.344 (287) | 3.443 / 3.443 (405) | 11.336 / 11.432 (2792) |
| stride_6 | 1200 (300) | 31/56 | 31/54 | +5 / -10, -0.093 [-0.200, +0.019] | 368 [299, 442] | 588 [471, 711] | 1.512 / 1.514 (363) | 0.438 / 0.627 (966) | 3.420 / 3.492 (1293) | 11.405 / 11.632 (8367) |

Stride 15 vs gate-off, GT events gate-off matched and stride 15 did not: lost 15 (0 fired within tolerance on a track attributed to another GT subject or none, 15 with no such event at all), shifted beyond 2.0 s 6 (gaps s: 2.97, 3.73, 10.70, 33.00, 34.53, 51.50), taken by another GT event 0. Matched events: signed time shift p50 0.233 s, |shift| p90 0.927 s.
