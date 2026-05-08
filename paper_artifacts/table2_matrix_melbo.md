# Table 2-MELBO — Cross-eval matrix (MELBO)

Each cell: best aligned-% achieved by a MELBO vector trained on the row's eval, evaluated on the column's eval.

| train \\ test | baseline | corrig | surv | power | wealth | self-aw | coord | myopic | row mean Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   corrig | 39 | **93** | 79 | 82 | 87 | 85 | 96 | 65 | +23.0 |
|     surv | 47 | 72 | **65** | 84 | 87 | 94 | 97 | 53 | +18.0 |
|    power | 61 | 90 | 61 | **85** | 85 | 93 | 99 | 54 | +20.1 |
|   wealth | 72 | 71 | 64 | 82 | **86** | 88 | 97 | 55 | +16.7 |
|  self-aw | 66 | 61 | 61 | 81 | 86 | **93** | 96 | 54 | +15.1 |
|    coord | 92 | 73 | 62 | 83 | 86 | 90 | **98** | 54 | +17.1 |
|   myopic | 49 | 64 | 57 | 78 | 80 | 85 | 94 | **60** | +13.1 |
| col mean Δ | — | +35.9 | +17.1 | +21.1 | +13.3 | +23.7 | +4.7 | +7.4 | +17.6 |
