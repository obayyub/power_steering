# Table 2 — Combined cross-eval matrix (all 4 methods per cell)

Each cell shows best aligned-% for CAA / PI / MELBO / DCT respectively, rendered top-to-bottom. **Bold** = cell winner. `★` = diagonal (specialist) cell.

| train \\ test | base | corrig | surv | power | wealth | self-aw | coord | myopic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **corrig** | 39 | CAA 64<br>PI 79<br>**MEL 93★**<br>DCT 73 | CAA 59<br>PI 66<br>**MEL 79**<br>DCT 65 | CAA 77<br>**PI 82**<br>MEL 82<br>DCT 81 | CAA 82<br>PI 84<br>MEL 87<br>**DCT 89** | CAA 78<br>PI 89<br>MEL 85<br>**DCT 90** | **CAA 98**<br>PI 97<br>MEL 96<br>DCT 97 | CAA 56<br>PI 54<br>**MEL 65**<br>DCT 55 |
| **surv** | 47 | CAA 58<br>PI 69<br>MEL 72<br>**DCT 80** | CAA 58<br>**PI 67★**<br>MEL 65<br>DCT 60 | CAA 62<br>PI 81<br>**MEL 84**<br>DCT 79 | CAA 76<br>PI 83<br>**MEL 87**<br>DCT 86 | CAA 68<br>PI 87<br>**MEL 94**<br>DCT 94 | CAA 93<br>**PI 98**<br>MEL 97<br>DCT 98 | CAA 53<br>PI 53<br>MEL 53<br>**DCT 59** |
| **power** | 61 | CAA 54<br>PI 74<br>**MEL 90**<br>DCT 77 | CAA 53<br>PI 59<br>MEL 61<br>**DCT 69** | CAA 69<br>PI 83<br>**MEL 85★**<br>DCT 81 | CAA 78<br>PI 83<br>**MEL 85**<br>DCT 84 | CAA 69<br>**PI 93**<br>MEL 93<br>DCT 80 | CAA 93<br>PI 97<br>**MEL 99**<br>DCT 97 | CAA 53<br>PI 53<br>MEL 54<br>**DCT 55** |
| **wealth** | 72 | CAA 55<br>PI 77<br>MEL 71<br>**DCT 79** | CAA 51<br>PI 64<br>MEL 64<br>**DCT 72** | CAA 74<br>**PI 85**<br>MEL 82<br>DCT 85 | CAA 79<br>**PI 86★**<br>MEL 86<br>DCT 86 | CAA 71<br>**PI 91**<br>MEL 88<br>DCT 90 | CAA 93<br>**PI 98**<br>MEL 97<br>DCT 97 | CAA 52<br>**PI 59**<br>MEL 55<br>DCT 54 |
| **self-aw** | 66 | CAA 48<br>**PI 69**<br>MEL 61<br>DCT 61 | CAA 53<br>PI 64<br>MEL 61<br>**DCT 66** | CAA 69<br>**PI 84**<br>MEL 81<br>DCT 82 | CAA 80<br>**PI 89**<br>MEL 86<br>DCT 89 | CAA 74<br>PI 95<br>MEL 93<br>**DCT 96★** | CAA 96<br>**PI 97**<br>MEL 96<br>DCT 96 | CAA 52<br>PI 53<br>MEL 54<br>**DCT 55** |
| **coord** | 92 | CAA 52<br>PI 71<br>**MEL 73**<br>DCT 70 | CAA 53<br>PI 59<br>**MEL 62**<br>DCT 61 | CAA 70<br>PI 80<br>**MEL 83**<br>DCT 83 | CAA 78<br>PI 84<br>MEL 86<br>**DCT 89** | CAA 67<br>**PI 93**<br>MEL 90<br>DCT 91 | CAA 93<br>**PI 98★**<br>MEL 98<br>DCT 98 | CAA 52<br>PI 54<br>MEL 54<br>**DCT 57** |
| **myopic** | 49 | CAA 50<br>PI 63<br>**MEL 64**<br>DCT 63 | CAA 57<br>**PI 64**<br>MEL 57<br>DCT 57 | CAA 63<br>PI 74<br>MEL 78<br>**DCT 80** | CAA 75<br>**PI 82**<br>MEL 80<br>DCT 81 | CAA 68<br>**PI 91**<br>MEL 85<br>DCT 82 | CAA 93<br>**PI 95**<br>MEL 94<br>DCT 95 | **CAA 79★**<br>PI 56<br>MEL 60<br>DCT 63 |

**Cell-winner counts** (ties split equally; 49 cells total, 7 diagonal):

| Method | Total wins | Diagonal (specialist) | Off-diagonal (generalist) |
|---|---:|---:|---:|
| CAA | 2.0 | 1.0 / 7 | 1.0 / 42 |
| PI | 14.7 | 1.7 / 7 | 13.0 / 42 |
| MELBO | 14.7 | 2.7 / 7 | 12.0 / 42 |
| DCT | 17.7 | 1.7 / 7 | 16.0 / 42 |

_10 cells had ties._
