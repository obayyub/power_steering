# Table 2-PI — Cross-eval matrix (PI)

Each cell: best aligned-% achieved by a PI vector trained on the row's eval, evaluated on the column's eval.

| train \\ test | baseline | corrig | surv | power | wealth | self-aw | coord | myopic | row mean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   corrig | 39 | **79** | 66 | 82 | 84 | 89 | 97 | 54 | +17.9 |
|     surv | 47 | 69 | **67** | 81 | 83 | 87 | 98 | 53 | +16.0 |
|    power | 61 | 74 | 59 | **83** | 83 | 93 | 97 | 53 | +16.6 |
|   wealth | 72 | 77 | 64 | 85 | **86** | 91 | 98 | 59 | +19.1 |
|  self-aw | 66 | 69 | 64 | 84 | 89 | **95** | 97 | 53 | +17.9 |
|    coord | 92 | 71 | 59 | 80 | 84 | 93 | **98** | 54 | +16.1 |
|   myopic | 49 | 63 | 64 | 74 | 82 | 91 | 95 | **56** | +14.1 |
| col mean Δ | — | +32.7 | +16.3 | +20.3 | +12.4 | +25.3 | +5.1 | +5.6 | +16.8 |
