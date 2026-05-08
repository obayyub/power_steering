# Per-train-eval × test-eval matrix — aligned direction (Qwen3-14B)

Rows: train eval × method (CAA / MELBO / PI). Each method's row uses
that method's best generalist (vector with highest mean alignment
shift across all 7 cols among row-trained vectors). Cell value:
`shift v<idx>@<scale>`. **Bold** = method that wins this (train, test) cell.

Sources: qwen3_14b_train_coordinate-other-ais, qwen3_14b_train_corrigible-neutral-HHH, qwen3_14b_train_myopic-reward, qwen3_14b_train_power-seeking-inclination, qwen3_14b_train_self-awareness-general-ai, qwen3_14b_train_survival-instinct, qwen3_14b_train_wealth-seeking-inclination

| Train | Method | coord-other | corrigible | myopic | power-seek | self-aware | survival | wealth-seek |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coord-other | CAA | +1 v0@-5 | +13 v0@-25 | **+3 v0@-10** | +9 v0@+25 | +1 v0@-2 | +6 v0@-10 | +6 v0@+5 |
|  | MELBO | **+5 v1@+5** | **+34 v1@-25** | +3 v1@-10 | +15 v1@+10 | +19 v1@+10 | +7 v1@-25 | **+12 v1@+5** |
|  | PI | +2 v0@+2 | +25 v0@+25 | +2 v0@-1 | **+18 v0@+10** | **+27 v0@+25** | **+10 v0@+25** | +11 v0@+25 |
| corrigible | CAA | **+6 v0@-25** | +25 v0@+25 | **+7 v0@+25** | +16 v0@-25 | +12 v0@-25 | +12 v0@+25 | +10 v0@-25 |
|  | MELBO | +2 v11@-25 | **+54 v11@+25** | +3 v11@-1 | **+20 v11@-25** | +15 v11@-25 | **+32 v11@+25** | **+15 v11@-25** |
|  | PI | +1 v10@-1 | +40 v10@-25 | +2 v10@-25 | +17 v10@+25 | **+23 v10@+25** | +6 v10@+5 | +5 v10@+10 |
| myopic | CAA | **+1 v0@-10** | +11 v0@+25 | **+30 v0@+25** | +2 v0@-10 | +2 v0@-10 | **+10 v0@+25** | +3 v0@+2 |
|  | MELBO | +1 v2@-2 | +15 v2@+25 | +3 v2@-1 | +7 v2@+5 | **+19 v2@+10** | +10 v2@-2 | +8 v2@+2 |
|  | PI | +1 v7@-10 | **+24 v7@-25** | +2 v7@-2 | **+13 v7@+25** | +18 v7@+25 | +3 v7@-25 | **+10 v7@+25** |
| power-seek | CAA | +1 v0@+1 | +15 v0@-25 | **+4 v0@-25** | +8 v0@+25 | +3 v0@+1 | +6 v0@-10 | +6 v0@+5 |
|  | MELBO | **+7 v10@-25** | **+51 v10@+25** | +3 v10@-10 | **+22 v10@-25** | +25 v10@-25 | **+8 v10@+25** | +9 v10@-25 |
|  | PI | +4 v0@-10 | +11 v0@+25 | +1 v0@+1 | +22 v0@-25 | **+27 v0@-25** | +8 v0@-10 | **+11 v0@-25** |
| self-aware | CAA | **+4 v0@+25** | +9 v0@-25 | **+3 v0@-10** | +8 v0@+10 | +8 v0@+10 | **+6 v0@+25** | +8 v0@+25 |
|  | MELBO | +3 v8@+25 | **+22 v8@-25** | +1 v8@-10 | +20 v8@+25 | +23 v8@+25 | +4 v8@+10 | +12 v8@+10 |
|  | PI | +4 v0@-25 | +4 v0@+25 | +1 v0@+1 | **+23 v0@-25** | **+29 v0@-25** | +5 v0@+2 | **+17 v0@-25** |
| survival | CAA | +1 v0@-5 | +19 v0@+25 | **+4 v0@-10** | +1 v0@-10 | +2 v0@+1 | **+11 v0@+10** | +4 v0@-5 |
|  | MELBO | **+5 v3@-10** | +21 v3@+10 | +4 v3@+2 | **+18 v3@-25** | **+24 v3@-25** | +8 v3@-5 | **+12 v3@-10** |
|  | PI | +5 v0@-10 | **+23 v0@+25** | +4 v0@+5 | +14 v0@-5 | +21 v0@-25 | +5 v0@+10 | +11 v0@-5 |
| wealth-seek | CAA | +1 v0@+1 | +16 v0@-25 | +3 v0@-10 | +13 v0@+25 | +5 v0@+10 | +4 v0@-25 | +7 v0@+10 |
|  | MELBO | +5 v11@+10 | **+32 v11@-25** | **+6 v11@+10** | +19 v11@+10 | +17 v11@+10 | +2 v11@+1 | +10 v11@+10 |
|  | PI | **+6 v0@-10** | +14 v0@+25 | +4 v0@+1 | **+21 v0@-25** | **+25 v0@-25** | **+9 v0@+2** | **+14 v0@-25** |
