# Table 2-CAA — Cross-eval matrix (CAA)

Each cell: best aligned-% achieved by a CAA vector trained on the row's eval, evaluated on the column's eval.

| train \\ test | baseline | corrig | surv | power | wealth | self-aw | coord | myopic | row mean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   corrig | 39 | **64** | 59 | 77 | 82 | 78 | 98 | 56 | +12.6 |
|     surv | 47 | 58 | **58** | 62 | 76 | 68 | 93 | 53 | +6.0 |
|    power | 61 | 54 | 53 | **69** | 78 | 69 | 93 | 53 | +6.1 |
|   wealth | 72 | 55 | 51 | 74 | **79** | 71 | 93 | 52 | +7.0 |
|  self-aw | 66 | 48 | 53 | 69 | 80 | **74** | 96 | 52 | +6.6 |
|    coord | 92 | 52 | 53 | 70 | 78 | 67 | **93** | 52 | +5.6 |
|   myopic | 49 | 50 | 57 | 63 | 75 | 68 | 93 | **79** | +8.4 |
| col mean Δ | — | +15.4 | +7.9 | +8.1 | +6.3 | +4.7 | +2.1 | +7.7 | +7.5 |
