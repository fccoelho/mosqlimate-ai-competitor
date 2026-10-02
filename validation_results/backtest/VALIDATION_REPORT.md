# IMDC Validation Report

Generated: 2026-10-02 13:27 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts.

## Chikungunya — mean WIS per model (tests with actuals)

| model        |    wis |   skill_vs_naive |   coverage_50 |
|:-------------|-------:|-----------------:|--------------:|
| timesfm      |  82.55 |             0.38 |          0.39 |
| ens_qavg     |  91.11 |             0.32 |          0.37 |
| ens_median   |  91.26 |             0.32 |          0.36 |
| xgb_direct   |  95.31 |             0.29 |          0.34 |
| lgbm_direct  | 100.19 |             0.25 |          0.31 |
| loglin_trend | 110.78 |             0.17 |          0.36 |
| seas_naive   | 133.6  |             0    |          0.29 |

### Best model per state (chikungunya)


timesfm: 14, loglin_trend: 4, seas_naive: 2, xgb_direct: 2, lgbm_direct: 2, ens_median: 1, ens_qavg: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.7 |    3.6 |    1.3 |   1.6 |
| AL      |   59.7 |   53.4 |   41.7 |  27.9 |
| AM      |    1.6 |    1.4 |    1.2 |   0.8 |
| AP      |    0.4 |    5.2 |    1.5 |   0.7 |
| BA      |  113.6 |  144.6 |  132.2 |  96.9 |
| CE      |  412   |  155.4 |   20.1 |  23   |
| DF      |    3.8 |    5.4 |    2.9 |   3.7 |
| GO      |   25   |  120.1 |   81.1 | 142.8 |
| MA      |   15.5 |   29.6 |    7.4 |   5.6 |
| MG      | 1470.6 | 1171   | 1641.6 | 414   |
| MS      |   86.4 |   39.2 |  113.8 |  76.9 |
| MT      |    3.2 |  368.7 |  468.1 | 711.9 |
| PA      |    5   |    9   |    4.5 |   3.6 |
| PB      |  245.6 |   52.1 |   18.7 |   7.6 |
| PE      |  299.3 |   56.2 |   24   |  30.9 |
| PI      |   64.3 |   65.9 |   13.2 |  10.7 |
| PR      |   49.2 |   21.7 |   59.1 |  59.6 |
| RJ      |   35.7 |   49.2 |   29.8 |  36.4 |
| RN      |  105.3 |   56.2 |   32.3 |  27.3 |
| RO      |    0.9 |    3   |   81.3 |  34.4 |
| RR      |    0.6 |    0.6 |    0.6 |   0.5 |
| RS      |    1.5 |    2.8 |    4.1 |   5.1 |
| SC      |    4.5 |    3.5 |    8.3 |   6.5 |
| SE      |   29   |   32.8 |   10.4 |   5.3 |
| SP      |   36.3 |  106.9 |   81.1 | 130.2 |
| TO      |   42.4 |   46.1 |    6.1 |   4.7 |

## Dengue — mean WIS per model (tests with actuals)

| model        |     wis |   skill_vs_naive |   coverage_50 |
|:-------------|--------:|-----------------:|--------------:|
| ens_median   | 1369.94 |             0.38 |          0.37 |
| ens_qavg     | 1384.28 |             0.37 |          0.37 |
| xgb_direct   | 1392.64 |             0.37 |          0.34 |
| loglin_trend | 1410.84 |             0.36 |          0.38 |
| lgbm_direct  | 1426.46 |             0.35 |          0.33 |
| timesfm      | 1473.68 |             0.33 |          0.36 |
| seas_naive   | 2209.05 |             0    |          0.39 |

### Best model per state (dengue)


timesfm: 8, loglin_trend: 6, lgbm_direct: 5, xgb_direct: 3, ens_qavg: 2, ens_median: 2

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   33.5 |    70   |    60   |    66.5 |
| AL      |  189.9 |   127.7 |    56.4 |    38.5 |
| AM      |   25.8 |    62.7 |    26.4 |    54.3 |
| AP      |   12.4 |   192.1 |    63.9 |    49.5 |
| BA      |  300.8 |  3471.9 |  1332.2 |   470.4 |
| CE      |  290.9 |   103.4 |    50   |   139.8 |
| DF      |  271.2 |  4240.7 |  1823   |   202.1 |
| GO      |  942.2 |  3496.3 |  1495.2 |   645.4 |
| MA      |   25.2 |   111   |    41.7 |   115.9 |
| MG      | 5106.6 | 23096.9 | 11288.3 |  2173.2 |
| MS      |  430.4 |   346.7 |   157.4 |   160.5 |
| MT      |  132.6 |   190.1 |   167.7 |   276.8 |
| PA      |   23.8 |   255.2 |    88.3 |    89.4 |
| PB      |  224.4 |    94   |    60.9 |    43.8 |
| PE      |  201.4 |   245.2 |   142.1 |    90.2 |
| PI      |  205.5 |   110.1 |    39.7 |   104.3 |
| PR      | 1366.2 |  5333.4 |  6028   |  2033.7 |
| RJ      |  486.2 |  4809   |  2093.5 |   483.1 |
| RN      |  191.2 |   157.9 |    62.5 |    46.1 |
| RO      |   82.2 |   100.9 |    26.9 |    18.7 |
| RR      |    2.5 |     9.2 |     3.3 |     3.2 |
| RS      |  471.8 |  2465.8 |  1380.1 |  1367.9 |
| SC      | 1111.2 |  2506.9 |  4370.8 |  2341.5 |
| SE      |   18.5 |    24.7 |    20.8 |    10.7 |
| SP      | 1903.5 | 24720.9 | 11694.3 | 13940.7 |
| TO      |  185.5 |    47.2 |    31.9 |   282.8 |

## Selected model per state (skill-gated)

**chikungunya**: timesfm: 14, loglin_trend: 4, seas_naive: 2, xgb_direct: 2, lgbm_direct: 2, ens_median: 1, ens_qavg: 1

**dengue**: timesfm: 8, loglin_trend: 6, lgbm_direct: 5, xgb_direct: 3, ens_qavg: 2, ens_median: 2

| state   | disease     | selected     |   mean_wis | fallback   |
|:--------|:------------|:-------------|-----------:|:-----------|
| AC      | chikungunya | timesfm      |        1.6 | False      |
| AC      | dengue      | loglin_trend |       45.8 | False      |
| AL      | chikungunya | timesfm      |       22.9 | False      |
| AL      | dengue      | timesfm      |       60.6 | False      |
| AM      | chikungunya | timesfm      |        1.1 | False      |
| AM      | dengue      | loglin_trend |       35.5 | False      |
| AP      | chikungunya | timesfm      |        1.8 | False      |
| AP      | dengue      | timesfm      |       67.9 | False      |
| BA      | chikungunya | ens_median   |       82.5 | False      |
| BA      | dengue      | lgbm_direct  |     1233.1 | False      |
| CE      | chikungunya | timesfm      |       59.1 | False      |
| CE      | dengue      | ens_qavg     |      125.8 | False      |
| DF      | chikungunya | timesfm      |        3.3 | False      |
| DF      | dengue      | xgb_direct   |     1420   | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1304.5 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       60.5 | False      |
| MG      | chikungunya | ens_qavg     |     1077.1 | False      |
| MG      | dengue      | lgbm_direct  |     9164   | False      |
| MS      | chikungunya | timesfm      |       73.1 | False      |
| MS      | dengue      | ens_median   |      242.1 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | xgb_direct   |      159.3 | False      |
| PA      | chikungunya | xgb_direct   |        3.8 | False      |
| PA      | dengue      | loglin_trend |      104.9 | False      |
| PB      | chikungunya | timesfm      |       41.8 | False      |
| PB      | dengue      | timesfm      |       72.4 | False      |
| PE      | chikungunya | timesfm      |       58   | False      |
| PE      | dengue      | timesfm      |      142.1 | False      |
| PI      | chikungunya | timesfm      |       31.2 | False      |
| PI      | dengue      | ens_median   |       90.2 | False      |
| PR      | chikungunya | timesfm      |       24.8 | False      |
| PR      | dengue      | lgbm_direct  |     2963.9 | False      |
| RJ      | chikungunya | xgb_direct   |       20.3 | False      |
| RJ      | dengue      | lgbm_direct  |     1744.7 | False      |
| RN      | chikungunya | timesfm      |       38.7 | False      |
| RN      | dengue      | timesfm      |       74.3 | False      |
| RO      | chikungunya | loglin_trend |       26.7 | False      |
| RO      | dengue      | timesfm      |       50   | False      |
| RR      | chikungunya | lgbm_direct  |        0.5 | False      |
| RR      | dengue      | ens_qavg     |        4.2 | False      |
| RS      | chikungunya | loglin_trend |        3   | False      |
| RS      | dengue      | xgb_direct   |     1227.1 | False      |
| SC      | chikungunya | lgbm_direct  |        4.7 | False      |
| SC      | dengue      | lgbm_direct  |     1777.9 | False      |
| SE      | chikungunya | seas_naive   |       13.9 | True       |
| SE      | dengue      | timesfm      |       15.8 | False      |
| SP      | chikungunya | timesfm      |       50.2 | False      |
| SP      | dengue      | timesfm      |     8849.3 | False      |
| TO      | chikungunya | timesfm      |       18.3 | False      |
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
