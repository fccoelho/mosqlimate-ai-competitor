# IMDC Validation Report

Generated: 2026-10-01 14:14 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts.

## Chikungunya — mean WIS per model (tests with actuals)

| model        |    wis |   skill_vs_naive |   coverage_50 |
|:-------------|-------:|-----------------:|--------------:|
| ens_qavg     | 102.81 |             0.23 |          0.34 |
| ens_median   | 106.51 |             0.2  |          0.3  |
| lgbm_direct  | 108.03 |             0.19 |          0.3  |
| xgb_direct   | 109.07 |             0.18 |          0.3  |
| loglin_trend | 110.78 |             0.17 |          0.36 |
| seas_naive   | 133.6  |             0    |          0.29 |

### Best model per state (chikungunya)


loglin_trend: 11, ens_median: 6, ens_qavg: 4, seas_naive: 3, lgbm_direct: 1, xgb_direct: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.7 |    4.1 |    1.3 |   1.9 |
| AL      |   65.2 |   60.6 |   56   |  36.1 |
| AM      |    1.6 |    1.5 |    1.1 |   0.8 |
| AP      |    0.4 |    5   |    1.5 |   0.9 |
| BA      |  110.1 |  162.9 |  157.8 | 133.4 |
| CE      |  488.1 |  199.6 |   20.3 |  20.2 |
| DF      |    3.6 |    6.5 |    3.3 |   4.3 |
| GO      |   27.8 |  120.5 |  106.4 | 137.8 |
| MA      |   15.7 |   37.5 |    9.2 |   6.7 |
| MG      | 1416.4 | 1163.7 | 2051.4 | 394   |
| MS      |   87.5 |   46.6 |  116.2 |  71.2 |
| MT      |    3.1 |  370.1 |  459.6 | 925.1 |
| PA      |    5.6 |   10.9 |    4.2 |   3.8 |
| PB      |  317.3 |   62.2 |   17.4 |   7.6 |
| PE      |  373.7 |   59.5 |   23.5 |  30.9 |
| PI      |   66.9 |   88.4 |   10.2 |   6.6 |
| PR      |   49.6 |   28   |   70.5 |  99.6 |
| RJ      |   19.3 |   46.6 |   27.1 |  39.2 |
| RN      |  132.3 |   61.8 |   30.8 |  24.1 |
| RO      |    1   |    3.1 |   81.5 |  42.9 |
| RR      |    0.6 |    0.5 |    0.6 |   0.5 |
| RS      |    1.5 |    3.1 |    4.2 |   5.8 |
| SC      |    4.5 |    3.7 |    8   |   8.1 |
| SE      |   41.4 |   27.8 |    7.6 |   5.5 |
| SP      |   43.2 |  136.3 |   83.4 | 168.5 |
| TO      |   44.3 |   55.5 |    6.9 |   4.1 |

## Dengue — mean WIS per model (tests with actuals)

| model        |     wis |   skill_vs_naive |   coverage_50 |
|:-------------|--------:|-----------------:|--------------:|
| loglin_trend | 1410.84 |             0.36 |          0.38 |
| ens_qavg     | 1480.82 |             0.33 |          0.37 |
| ens_median   | 1543.54 |             0.3  |          0.35 |
| lgbm_direct  | 1561.67 |             0.29 |          0.34 |
| xgb_direct   | 1562.33 |             0.29 |          0.34 |
| seas_naive   | 2209.05 |             0    |          0.39 |

### Best model per state (dengue)


loglin_trend: 16, ens_qavg: 9, xgb_direct: 1

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   29.6 |    81   |    55.6 |   106.6 |
| AL      |  241.8 |   132.4 |    72.6 |    45   |
| AM      |   25.1 |    72.8 |    28.4 |    49.5 |
| AP      |   12.4 |   192.8 |    70.6 |    61.3 |
| BA      |  263.6 |  3410.7 |  1610.3 |   468.2 |
| CE      |  387.6 |    93.3 |    42.5 |   167.9 |
| DF      |  291.8 |  4052.7 |  2008.4 |   197   |
| GO      | 1032.6 |  3159.6 |  1603.6 |   727.3 |
| MA      |   28.9 |   106.8 |    42.3 |   111.8 |
| MG      | 4691.5 | 21620.5 | 13337.4 |  2736.9 |
| MS      |  416.5 |   465.4 |   101.9 |    91.6 |
| MT      |  133.8 |   154.1 |   162.5 |   360.4 |
| PA      |   23.1 |   240.1 |    84.3 |   143.5 |
| PB      |  344.2 |    85.2 |    54.9 |    44   |
| PE      |  233   |   311.6 |   121.6 |    93.7 |
| PI      |  231.3 |   115.9 |    54.1 |   118.1 |
| PR      | 1266.1 |  4739.7 |  5789.3 |  2660.1 |
| RJ      |  473.4 |  4751.9 |  2396   |   634   |
| RN      |  243.1 |   160.6 |    58.3 |    55.9 |
| RO      |   84.5 |   115.3 |    30.5 |    15.8 |
| RR      |    2.5 |     9.7 |     3.1 |     3.5 |
| RS      |  468.6 |  2541.3 |  1420.3 |  1881.8 |
| SC      | 1060.5 |  2604.6 |  4731.2 |   995.9 |
| SE      |   20.2 |    28   |    28.6 |    13.3 |
| SP      | 1587.7 | 27829.7 | 12943.9 | 19405.5 |
| TO      |  222.3 |    41.5 |    15.5 |   326.9 |

## Selected model per state (skill-gated)

**chikungunya**: loglin_trend: 11, ens_median: 6, ens_qavg: 4, seas_naive: 3, lgbm_direct: 1, xgb_direct: 1

**dengue**: loglin_trend: 16, ens_qavg: 9, xgb_direct: 1

| state   | disease     | selected     |   mean_wis | fallback   |
|:--------|:------------|:-------------|-----------:|:-----------|
| AC      | chikungunya | loglin_trend |        1.9 | False      |
| AC      | dengue      | loglin_trend |       45.8 | False      |
| AL      | chikungunya | loglin_trend |       39.4 | False      |
| AL      | dengue      | loglin_trend |       82.8 | False      |
| AM      | chikungunya | ens_qavg     |        1.2 | False      |
| AM      | dengue      | loglin_trend |       35.5 | False      |
| AP      | chikungunya | lgbm_direct  |        1.8 | False      |
| AP      | dengue      | loglin_trend |       72.6 | False      |
| BA      | chikungunya | ens_median   |      108.2 | False      |
| BA      | dengue      | ens_qavg     |     1271.6 | False      |
| CE      | chikungunya | loglin_trend |      149.8 | False      |
| CE      | dengue      | loglin_trend |      126.8 | False      |
| DF      | chikungunya | ens_qavg     |        4.1 | False      |
| DF      | dengue      | ens_qavg     |     1429.3 | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1304.5 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       60.5 | False      |
| MG      | chikungunya | ens_qavg     |     1159.9 | False      |
| MG      | dengue      | ens_qavg     |     9547.2 | False      |
| MS      | chikungunya | ens_qavg     |       76.5 | False      |
| MS      | dengue      | ens_qavg     |      237.6 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | ens_qavg     |      187.8 | False      |
| PA      | chikungunya | ens_median   |        4.9 | False      |
| PA      | dengue      | loglin_trend |      104.9 | False      |
| PB      | chikungunya | xgb_direct   |       83.6 | False      |
| PB      | dengue      | loglin_trend |       88.8 | False      |
| PE      | chikungunya | seas_naive   |      102.1 | True       |
| PE      | dengue      | loglin_trend |      154.6 | False      |
| PI      | chikungunya | loglin_trend |       36.4 | False      |
| PI      | dengue      | loglin_trend |       98.2 | False      |
| PR      | chikungunya | loglin_trend |       42.1 | False      |
| PR      | dengue      | ens_qavg     |     3185.5 | False      |
| RJ      | chikungunya | ens_median   |       22.3 | False      |
| RJ      | dengue      | xgb_direct   |     1879.9 | False      |
| RN      | chikungunya | ens_median   |       46.5 | False      |
| RN      | dengue      | loglin_trend |       81.7 | False      |
| RO      | chikungunya | loglin_trend |       26.7 | False      |
| RO      | dengue      | ens_qavg     |       57.1 | False      |
| RR      | chikungunya | ens_median   |        0.5 | False      |
| RR      | dengue      | ens_qavg     |        4.5 | False      |
| RS      | chikungunya | loglin_trend |        3   | False      |
| RS      | dengue      | loglin_trend |     1348.8 | False      |
| SC      | chikungunya | loglin_trend |        5.1 | False      |
| SC      | dengue      | ens_qavg     |     2157.5 | False      |
| SE      | chikungunya | seas_naive   |       13.9 | True       |
| SE      | dengue      | loglin_trend |       18.5 | False      |
| SP      | chikungunya | loglin_trend |       84.4 | False      |
| SP      | dengue      | loglin_trend |    12676.2 | False      |
| TO      | chikungunya | ens_median   |       23.9 | False      |
| TO      | dengue      | loglin_trend |      112.8 | False      |

## Forecasts vs observed

Per-state panels: observed training tail, the selected model's
calibrated median with 50%/95% bands, the 15-week unobserved gap
(shaded), and the observed target season. WIS shown per panel
when the season has observed data.

### AC — chikungunya

![AC/chikungunya](plots/chikungunya/AC.png)

### AC — dengue

![AC/dengue](plots/dengue/AC.png)

### AL — chikungunya

![AL/chikungunya](plots/chikungunya/AL.png)

### AL — dengue

![AL/dengue](plots/dengue/AL.png)

### AM — chikungunya

![AM/chikungunya](plots/chikungunya/AM.png)

### AM — dengue

![AM/dengue](plots/dengue/AM.png)

### AP — chikungunya

![AP/chikungunya](plots/chikungunya/AP.png)

### AP — dengue

![AP/dengue](plots/dengue/AP.png)

### BA — chikungunya

![BA/chikungunya](plots/chikungunya/BA.png)

### BA — dengue

![BA/dengue](plots/dengue/BA.png)

### CE — chikungunya

![CE/chikungunya](plots/chikungunya/CE.png)

### CE — dengue

![CE/dengue](plots/dengue/CE.png)

### DF — chikungunya

![DF/chikungunya](plots/chikungunya/DF.png)

### DF — dengue

![DF/dengue](plots/dengue/DF.png)

### GO — chikungunya

![GO/chikungunya](plots/chikungunya/GO.png)

### GO — dengue

![GO/dengue](plots/dengue/GO.png)

### MA — chikungunya

![MA/chikungunya](plots/chikungunya/MA.png)

### MA — dengue

![MA/dengue](plots/dengue/MA.png)

### MG — chikungunya

![MG/chikungunya](plots/chikungunya/MG.png)

### MG — dengue

![MG/dengue](plots/dengue/MG.png)

### MS — chikungunya

![MS/chikungunya](plots/chikungunya/MS.png)

### MS — dengue

![MS/dengue](plots/dengue/MS.png)

### MT — chikungunya

![MT/chikungunya](plots/chikungunya/MT.png)

### MT — dengue

![MT/dengue](plots/dengue/MT.png)

### PA — chikungunya

![PA/chikungunya](plots/chikungunya/PA.png)

### PA — dengue

![PA/dengue](plots/dengue/PA.png)

### PB — chikungunya

![PB/chikungunya](plots/chikungunya/PB.png)

### PB — dengue

![PB/dengue](plots/dengue/PB.png)

### PE — chikungunya

![PE/chikungunya](plots/chikungunya/PE.png)

### PE — dengue

![PE/dengue](plots/dengue/PE.png)

### PI — chikungunya

![PI/chikungunya](plots/chikungunya/PI.png)

### PI — dengue

![PI/dengue](plots/dengue/PI.png)

### PR — chikungunya

![PR/chikungunya](plots/chikungunya/PR.png)

### PR — dengue

![PR/dengue](plots/dengue/PR.png)

### RJ — chikungunya

![RJ/chikungunya](plots/chikungunya/RJ.png)

### RJ — dengue

![RJ/dengue](plots/dengue/RJ.png)

### RN — chikungunya

![RN/chikungunya](plots/chikungunya/RN.png)

### RN — dengue

![RN/dengue](plots/dengue/RN.png)

### RO — chikungunya

![RO/chikungunya](plots/chikungunya/RO.png)

### RO — dengue

![RO/dengue](plots/dengue/RO.png)

### RR — chikungunya

![RR/chikungunya](plots/chikungunya/RR.png)

### RR — dengue

![RR/dengue](plots/dengue/RR.png)

### RS — chikungunya

![RS/chikungunya](plots/chikungunya/RS.png)

### RS — dengue

![RS/dengue](plots/dengue/RS.png)

### SC — chikungunya

![SC/chikungunya](plots/chikungunya/SC.png)

### SC — dengue

![SC/dengue](plots/dengue/SC.png)

### SE — chikungunya

![SE/chikungunya](plots/chikungunya/SE.png)

### SE — dengue

![SE/dengue](plots/dengue/SE.png)

### SP — chikungunya

![SP/chikungunya](plots/chikungunya/SP.png)

### SP — dengue

![SP/dengue](plots/dengue/SP.png)

### TO — chikungunya

![TO/chikungunya](plots/chikungunya/TO.png)

### TO — dengue

![TO/dengue](plots/dengue/TO.png)
