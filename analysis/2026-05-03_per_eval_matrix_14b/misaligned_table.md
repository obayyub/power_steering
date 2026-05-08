# Per-train-eval × test-eval matrix — misaligned direction (Qwen3-14B)

Rows: train eval × method (CAA / MELBO / PI). Each method's row uses
that method's best generalist (vector with highest mean alignment
shift across all 7 cols among row-trained vectors). Cell value:
`shift v<idx>@<scale>`. **Bold** = method that wins this (train, test) cell.

Sources: qwen3_14b_train_coordinate-other-ais, qwen3_14b_train_corrigible-neutral-HHH, qwen3_14b_train_myopic-reward, qwen3_14b_train_power-seeking-inclination, qwen3_14b_train_self-awareness-general-ai, qwen3_14b_train_survival-instinct, qwen3_14b_train_wealth-seeking-inclination

| Train | Method | coord-other | corrigible | myopic | power-seek | self-aware | survival | wealth-seek |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coord-other | CAA | -13 v0@-25 | -3 v0@-10 | **-18 v0@+25** | -7 v0@-25 | -10 v0@-25 | -3 v0@+25 | -22 v0@-25 |
|  | MELBO | **-61 v1@-25** | **-28 v1@+25** | -11 v1@+5 | -11 v1@-25 | **-30 v1@-25** | **-16 v1@+25** | **-27 v1@-25** |
|  | PI | -45 v1@+25 | -27 v1@-25 | -14 v1@-10 | **-12 v1@+5** | -17 v1@+25 | -12 v1@-25 | -20 v1@+10 |
| corrigible | CAA | -6 v0@+25 | **-20 v0@-25** | -15 v0@-25 | -8 v0@+25 | **-21 v0@+25** | **-14 v0@-25** | -11 v0@+25 |
|  | MELBO | -39 v1@-25 | -17 v1@+25 | -12 v1@-5 | -7 v1@-25 | -15 v1@-25 | -8 v1@+10 | **-22 v1@-25** |
|  | PI | **-40 v0@-25** | -15 v0@+25 | **-36 v0@+25** | **-12 v0@-25** | -17 v0@-25 | -4 v0@+10 | -21 v0@-10 |
| myopic | CAA | -6 v0@+25 | -3 v0@-5 | -18 v0@-25 | -3 v0@+10 | -17 v0@+25 | -1 v0@-2 | -8 v0@-25 |
|  | MELBO | **-43 v2@-25** | **-5 v2@-2** | **-22 v2@+25** | **-12 v2@-25** | **-18 v2@-25** | +0 v2@+0 | **-22 v2@-10** |
|  | PI | -42 v3@-25 | -5 v3@-2 | -18 v3@+25 | -12 v3@-10 | -18 v3@-25 | **-2 v3@+25** | -22 v3@-25 |
| power-seek | CAA | -10 v0@-25 | -2 v0@+10 | **-17 v0@+25** | -9 v0@-25 | -16 v0@-25 | -1 v0@+1 | -17 v0@-25 |
|  | MELBO | **-49 v10@+25** | **-29 v10@-25** | -16 v10@+25 | **-20 v10@+25** | **-35 v10@+25** | **-2 v10@-25** | **-36 v10@+25** |
|  | PI | -42 v4@-25 | -23 v4@+25 | -9 v4@+25 | -19 v4@-25 | -30 v4@-25 | -2 v4@+5 | -29 v4@-25 |
| self-aware | CAA | -5 v0@-25 | -5 v0@+10 | -16 v0@+25 | -7 v0@-25 | -26 v0@-25 | -1 v0@+1 | -14 v0@-25 |
|  | MELBO | **-42 v2@+25** | -1 v2@+1 | **-19 v2@-25** | -11 v2@+25 | -21 v2@+10 | -4 v2@+2 | -22 v2@+25 |
|  | PI | -23 v5@+25 | **-7 v5@-5** | -5 v5@-25 | **-17 v5@+25** | **-35 v5@+10** | **-8 v5@+10** | **-29 v5@+25** |
| survival | CAA | +0 v0@+0 | -13 v0@-25 | -5 v0@+10 | -2 v0@-25 | -12 v0@-25 | **-15 v0@-25** | -8 v0@+25 |
|  | MELBO | -34 v3@+25 | -16 v3@-25 | **-23 v3@-25** | -9 v3@+25 | **-36 v3@+25** | -5 v3@-25 | -17 v3@+25 |
|  | PI | **-36 v0@+25** | **-26 v0@-25** | -18 v0@-10 | **-10 v0@+10** | -25 v0@+25 | -5 v0@-25 | **-19 v0@+10** |
| wealth-seek | CAA | -10 v0@-25 | -5 v0@+10 | -13 v0@+25 | -6 v0@-25 | -19 v0@-25 | -1 v0@-5 | -20 v0@-25 |
|  | MELBO | **-41 v0@+25** | -10 v0@-25 | **-28 v0@-25** | **-12 v0@+10** | -16 v0@+25 | -6 v0@-25 | -23 v0@+25 |
|  | PI | -24 v8@-25 | **-15 v8@+25** | -18 v8@-25 | -4 v8@-25 | **-41 v8@-25** | **-8 v8@-10** | **-27 v8@-25** |
