# Table 2-DCT — Cross-eval matrix (DCT)

Each cell: best aligned-% achieved by a DCT vector trained on the row's eval, evaluated on the column's eval.

| train \\ test | baseline | corrig | surv | power | wealth | self-aw | coord | myopic | row mean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   corrig | 39 | **73** | 65 | 81 | 89 | 90 | 97 | 55 | +17.7 |
|     surv | 47 | 80 | **60** | 79 | 86 | 94 | 98 | 59 | +18.6 |
|    power | 61 | 77 | 69 | **81** | 84 | 80 | 97 | 55 | +16.7 |
|   wealth | 72 | 79 | 72 | 85 | **86** | 90 | 97 | 54 | +19.6 |
|  self-aw | 66 | 61 | 66 | 82 | 89 | **96** | 96 | 55 | +17.0 |
|    coord | 92 | 70 | 61 | 83 | 89 | 91 | **98** | 57 | +17.6 |
|   myopic | 49 | 63 | 57 | 80 | 81 | 82 | 95 | **63** | +13.6 |
| col mean Δ | — | +32.9 | +17.3 | +20.6 | +14.3 | +23.0 | +4.9 | +7.9 | +17.2 |
