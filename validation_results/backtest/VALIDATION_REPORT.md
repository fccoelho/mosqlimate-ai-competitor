# IMDC Validation Report

Generated: 2026-10-03 16:33 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts.

## Chikungunya — mean WIS per model (tests with actuals)

| model        |    wis |   skill_vs_naive |   coverage_50 |
|:-------------|-------:|-----------------:|--------------:|
| xgb_direct   |  95.31 |             0.29 |          0.34 |
| ens_median   |  95.73 |             0.28 |          0.34 |
| ens_qavg     |  96.33 |             0.28 |          0.35 |
| lgbm_direct  | 100.19 |             0.25 |          0.31 |
| loglin_trend | 110.78 |             0.17 |          0.36 |
| seas_naive   | 133.6  |             0    |          0.29 |

### Best model per state (chikungunya)


loglin_trend: 8, lgbm_direct: 5, ens_median: 5, xgb_direct: 4, ens_qavg: 2, seas_naive: 2

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.7 |    3.9 |    1.3 |   1.7 |
| AL      |   66.1 |   60.1 |   48.8 |  31.5 |
| AM      |    1.6 |    1.5 |    1.3 |   0.8 |
| AP      |    0.4 |    5.1 |    1.5 |   0.8 |
| BA      |  112.1 |  156.6 |  153.3 | 105.8 |
| CE      |  468.3 |  188.6 |   23.3 |  25   |
| DF      |    3.8 |    5.7 |    3.4 |   3.7 |
| GO      |   26.6 |  116.5 |   86.1 | 142.2 |
| MA      |   14.7 |   32.5 |    8.7 |   5.6 |
| MG      | 1439.7 | 1091.8 | 1824.2 | 413   |
| MS      |   86.9 |   41.7 |  116.1 |  78.1 |
| MT      |    3.2 |  370.2 |  438.4 | 812   |
| PA      |    5.6 |   10.4 |    4.2 |   3.1 |
| PB      |  271.1 |   61.6 |   19.8 |   9   |
| PE      |  335.3 |   63.5 |   23.8 |  32.8 |
| PI      |   67.4 |   74.3 |   10.1 |   8.7 |
| PR      |   49.7 |   24.3 |   68.2 |  70.8 |
| RJ      |   17.2 |   45.1 |   26.6 |  38.8 |
| RN      |  118.2 |   64.2 |   30.7 |  24.3 |
| RO      |    0.9 |    3.1 |   81   |  37.4 |
| RR      |    0.6 |    0.5 |    0.6 |   0.5 |
| RS      |    1.5 |    3.2 |    4.1 |   4.9 |
| SC      |    4.8 |    3.6 |    7.9 |   6.8 |
| SE      |   31.6 |   32   |    9.4 |   5.3 |
| SP      |   41.4 |  133.1 |   84.8 | 145.2 |
| TO      |   43.2 |   50.6 |    7   |   4.7 |

## Dengue — mean WIS per model (tests with actuals)

| model        |     wis |   skill_vs_naive |   coverage_50 |
|:-------------|--------:|-----------------:|--------------:|
| ens_qavg     | 1389.4  |             0.37 |          0.37 |
| ens_median   | 1389.87 |             0.37 |          0.35 |
| xgb_direct   | 1392.64 |             0.37 |          0.34 |
| loglin_trend | 1410.84 |             0.36 |          0.38 |
| lgbm_direct  | 1426.46 |             0.35 |          0.33 |
| seas_naive   | 2209.05 |             0    |          0.39 |

### Best model per state (dengue)


loglin_trend: 12, lgbm_direct: 6, ens_qavg: 4, xgb_direct: 3, ens_median: 1

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   28.3 |    78   |    57.6 |    66.5 |
| AL      |  219.1 |   131.9 |    58.5 |    43.2 |
| AM      |   25.3 |    64.9 |    27.6 |    47.1 |
| AP      |   12.7 |   193.4 |    69.4 |    54.1 |
| BA      |  297.7 |  3443.2 |  1434.6 |   490.9 |
| CE      |  304.8 |    90.3 |    54.5 |   124.5 |
| DF      |  280.9 |  4083.1 |  1982.1 |   212.8 |
| GO      | 1081.4 |  3140.1 |  1538.8 |   705.1 |
| MA      |   25.6 |   108.6 |    42.7 |   116.4 |
| MG      | 5027.7 | 22159.5 | 12003   |  2356.7 |
| MS      |  410.9 |   401   |   115.2 |   117.3 |
| MT      |  136.2 |   160.7 |   149.4 |   270.5 |
| PA      |   22.9 |   249.5 |    89.1 |   104.3 |
| PB      |  271.2 |    90.9 |    54.3 |    46.3 |
| PE      |  216.6 |   266.7 |   142.8 |    89.7 |
| PI      |  224.5 |   112.8 |    37.8 |   104.2 |
| PR      | 1255.6 |  4887.5 |  4960.9 |  2430.5 |
| RJ      |  512.7 |  4748.7 |  2234.9 |   529.9 |
| RN      |  226.7 |   171.5 |    54   |    47.8 |
| RO      |   80.9 |   109.4 |    27.7 |    17.8 |
| RR      |    2.6 |     9.3 |     3.1 |     3.3 |
| RS      |  481.2 |  2390   |  1378.7 |  1272.2 |
| SC      | 1011.1 |  2591   |  3831.5 |  1070.8 |
| SE      |   19.7 |    27.2 |    20.6 |    10.8 |
| SP      | 1756.6 | 26557   | 12627.3 | 15692.9 |
| TO      |  214.9 |    49.3 |    18.4 |   279.3 |

## Selected model per state (skill-gated)

**chikungunya**: loglin_trend: 8, lgbm_direct: 5, ens_median: 5, xgb_direct: 4, ens_qavg: 2, seas_naive: 2

**dengue**: loglin_trend: 12, lgbm_direct: 6, ens_qavg: 4, xgb_direct: 3, ens_median: 1

| state   | disease     | selected     |   mean_wis | fallback   |
|:--------|:------------|:-------------|-----------:|:-----------|
| AC      | chikungunya | lgbm_direct  |        1.8 | False      |
| AC      | dengue      | loglin_trend |       45.8 | False      |
| AL      | chikungunya | loglin_trend |       39.4 | False      |
| AL      | dengue      | loglin_trend |       82.8 | False      |
| AM      | chikungunya | loglin_trend |        1.2 | False      |
| AM      | dengue      | loglin_trend |       35.5 | False      |
| AP      | chikungunya | lgbm_direct  |        1.9 | False      |
| AP      | dengue      | loglin_trend |       72.6 | False      |
| BA      | chikungunya | ens_median   |       90.9 | False      |
| BA      | dengue      | lgbm_direct  |     1233.1 | False      |
| CE      | chikungunya | lgbm_direct  |      137.4 | False      |
| CE      | dengue      | ens_qavg     |      126.7 | False      |
| DF      | chikungunya | ens_qavg     |        3.9 | False      |
| DF      | dengue      | xgb_direct   |     1420   | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1304.5 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       60.5 | False      |
| MG      | chikungunya | ens_qavg     |     1091.9 | False      |
| MG      | dengue      | lgbm_direct  |     9164   | False      |
| MS      | chikungunya | ens_median   |       76.8 | False      |
| MS      | dengue      | ens_qavg     |      243.3 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | xgb_direct   |      159.3 | False      |
| PA      | chikungunya | xgb_direct   |        3.8 | False      |
| PA      | dengue      | loglin_trend |      104.9 | False      |
| PB      | chikungunya | lgbm_direct  |       66.2 | False      |
| PB      | dengue      | loglin_trend |       88.8 | False      |
| PE      | chikungunya | xgb_direct   |       97.4 | False      |
| PE      | dengue      | loglin_trend |      154.6 | False      |
| PI      | chikungunya | ens_median   |       34.8 | False      |
| PI      | dengue      | ens_qavg     |       93.3 | False      |
| PR      | chikungunya | loglin_trend |       42.1 | False      |
| PR      | dengue      | lgbm_direct  |     2963.9 | False      |
| RJ      | chikungunya | ens_median   |       19.8 | False      |
| RJ      | dengue      | lgbm_direct  |     1744.7 | False      |
| RN      | chikungunya | xgb_direct   |       40.7 | False      |
| RN      | dengue      | loglin_trend |       81.7 | False      |
| RO      | chikungunya | loglin_trend |       26.7 | False      |
| RO      | dengue      | ens_median   |       54.4 | False      |
| RR      | chikungunya | ens_median   |        0.5 | False      |
| RR      | dengue      | lgbm_direct  |        4.3 | False      |
| RS      | chikungunya | loglin_trend |        3   | False      |
| RS      | dengue      | xgb_direct   |     1227.1 | False      |
| SC      | chikungunya | lgbm_direct  |        4.7 | False      |
| SC      | dengue      | lgbm_direct  |     1777.9 | False      |
| SE      | chikungunya | seas_naive   |       13.9 | True       |
| SE      | dengue      | ens_qavg     |       17.7 | False      |
| SP      | chikungunya | loglin_trend |       84.4 | False      |
| SP      | dengue      | loglin_trend |    12676.2 | False      |
| TO      | chikungunya | xgb_direct   |       21.6 | False      |
| TO      | dengue      | loglin_trend |      112.8 | False      |

## Per-state hyperparameter tuning

104 tuned configurations (random search on a held-out 67-week window, WIS criterion; cached under `hyperparams/`). Mean gain over defaults: 19.4%; tuning improved WIS for 102/104 configs.

| state   | disease     | model       |   tuned_wis |   default_wis |   gain_pct |   max_depth |   n_estimators |
|:--------|:------------|:------------|------------:|--------------:|-----------:|------------:|---------------:|
| AC      | dengue      | lgbm_direct |       31.88 |         62.64 |       49.1 |           7 |            244 |
| MA      | dengue      | xgb_direct  |       31.35 |         58.13 |       46.1 |           3 |            246 |
| CE      | dengue      | xgb_direct  |      313.13 |        551.37 |       43.2 |           3 |            246 |
| MA      | dengue      | lgbm_direct |       33.35 |         55.42 |       39.8 |           3 |            246 |
| AC      | dengue      | xgb_direct  |       44.28 |         69.98 |       36.7 |           7 |            244 |
| CE      | dengue      | lgbm_direct |      360.02 |        565.31 |       36.3 |           3 |            246 |
| MT      | dengue      | xgb_direct  |      209.47 |        321.7  |       34.9 |           3 |            246 |
| SE      | dengue      | xgb_direct  |       25.36 |         38.5  |       34.1 |           3 |            246 |
| AM      | dengue      | xgb_direct  |       29.2  |         43.83 |       33.4 |           3 |            246 |
| MG      | chikungunya | xgb_direct  |       26.85 |         40.24 |       33.3 |           3 |            150 |
| MG      | dengue      | xgb_direct  |      534.47 |        784.14 |       31.8 |           7 |            244 |
| SE      | dengue      | lgbm_direct |       26.72 |         39.01 |       31.5 |           3 |            246 |
| MA      | chikungunya | xgb_direct  |       17.51 |         25.22 |       30.6 |           3 |            246 |
| PA      | dengue      | xgb_direct  |       34.06 |         49.03 |       30.5 |           3 |            246 |
| RR      | chikungunya | lgbm_direct |        0.37 |          0.53 |       30.3 |           3 |            150 |
| AM      | dengue      | lgbm_direct |       27.26 |         38.72 |       29.6 |           3 |            246 |
| AL      | dengue      | lgbm_direct |      267.52 |        376.98 |       29   |           3 |            246 |
| BA      | chikungunya | xgb_direct  |       86.24 |        120.67 |       28.5 |           3 |            246 |
| PB      | dengue      | xgb_direct  |      245.23 |        342.35 |       28.4 |           3 |            246 |
| GO      | dengue      | xgb_direct  |     1635.57 |       2276.82 |       28.2 |           3 |            246 |

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
