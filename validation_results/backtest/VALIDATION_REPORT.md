# IMDC Validation Report

Generated: 2026-09-30 21:25 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts.

## Chikungunya — mean WIS per model (tests with actuals)

| model        |    wis |   skill_vs_naive |   coverage_50 |
|:-------------|-------:|-----------------:|--------------:|
| ens_qavg     |  98.41 |             0.26 |          0.35 |
| ens_median   |  98.99 |             0.26 |          0.33 |
| xgb_direct   |  99.24 |             0.26 |          0.33 |
| lgbm_direct  | 101.76 |             0.24 |          0.31 |
| loglin_trend | 110.78 |             0.17 |          0.36 |
| seas_naive   | 133.6  |             0    |          0.29 |

### Best model per state (chikungunya)


loglin_trend: 8, ens_median: 7, xgb_direct: 6, ens_qavg: 2, seas_naive: 2, lgbm_direct: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.7 |    4   |    1.3 |   1.9 |
| AL      |   62.9 |   60.5 |   47.9 |  31.3 |
| AM      |    1.6 |    1.5 |    1.2 |   0.8 |
| AP      |    0.4 |    5.1 |    1.5 |   0.8 |
| BA      |  113.5 |  155.1 |  153.5 | 102.7 |
| CE      |  445.5 |  175.1 |   22.2 |  26   |
| DF      |    3.7 |    5.5 |    3.6 |   3.7 |
| GO      |   26.3 |  119.3 |   88.7 | 139.1 |
| MA      |   15   |   34.3 |    7   |   5.6 |
| MG      | 1439.1 | 1204.8 | 1842.8 | 400.1 |
| MS      |   86.2 |   42.6 |  116.9 |  77.2 |
| MT      |    3.2 |  369.7 |  457.9 | 866.4 |
| PA      |    5.6 |   11.1 |    4.5 |   3.4 |
| PB      |  292.6 |   62.7 |   18.4 |   8.3 |
| PE      |  339.4 |   58.9 |   25.7 |  33.7 |
| PI      |   66.7 |   73.3 |   10.4 |   8.7 |
| PR      |   49.6 |   23   |   68.9 |  69.8 |
| RJ      |   16.6 |   50.3 |   26.8 |  40.7 |
| RN      |  127.5 |   65.6 |   31.2 |  24.1 |
| RO      |    0.9 |    3   |   81.1 |  38   |
| RR      |    0.6 |    0.5 |    0.6 |   0.5 |
| RS      |    1.5 |    3.1 |    4.3 |   5.2 |
| SC      |    4.5 |    3.4 |    7.9 |   6.7 |
| SE      |   32.3 |   30.6 |    8.8 |   4.9 |
| SP      |   41.4 |  130.9 |   81.4 | 150   |
| TO      |   42.4 |   50.8 |    7.6 |   5.1 |

## Dengue — mean WIS per model (tests with actuals)

| model        |     wis |   skill_vs_naive |   coverage_50 |
|:-------------|--------:|-----------------:|--------------:|
| ens_qavg     | 1522    |             0.4  |          0.36 |
| loglin_trend | 1522.88 |             0.4  |          0.39 |
| lgbm_direct  | 1543.43 |             0.39 |          0.34 |
| ens_median   | 1545.83 |             0.39 |          0.34 |
| xgb_direct   | 1566.41 |             0.38 |          0.32 |
| seas_naive   | 2527.72 |             0    |          0.38 |

### Best model per state (dengue)


loglin_trend: 12, lgbm_direct: 4, ens_median: 4, ens_qavg: 3, xgb_direct: 3

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   29.4 |    76.4 |    75.5 |    74.4 |
| AL      |  213.5 |   131.4 |    52   |    55   |
| AM      |   26.2 |    67   |    30.1 |    63.6 |
| AP      |   12.4 |   193.7 |    76.2 |    67   |
| BA      |  273.8 |  3450.2 |  1785.4 |   678.6 |
| CE      |  315.9 |    83.8 |    59.4 |   162.6 |
| DF      |  274.4 |  4099.2 |  2385.2 |   303.9 |
| GO      |  994.7 |  3250.5 |  1800.2 |   871.6 |
| MA      |   26.6 |   108.7 |    48.5 |   145.4 |
| MG      | 4922   | 21658.1 | 14503.9 |  4011.6 |
| MS      |  455.8 |   423.8 |   135.8 |   165.9 |
| MT      |  133.1 |   162.1 |   174.4 |   389.6 |
| PA      |   23.2 |   252.8 |    99.2 |   129.1 |
| PB      |  270.5 |    90   |    61.7 |    67.4 |
| PE      |  217   |   244.7 |   103.1 |   104.5 |
| PI      |  219.8 |   114   |    43.1 |   151.2 |
| PR      | 1275.8 |  4817.3 |  5543.2 |  3559   |
| RJ      |  536.1 |  4694.3 |  2457.2 |   665.4 |
| RN      |  222.4 |   168.8 |    53.3 |    72.3 |
| RO      |   83.4 |   106.4 |    35   |    21   |
| RR      |    2.8 |     9.2 |     2.8 |     3.4 |
| RS      |  486.6 |  2406.5 |  1559.9 |  1744.6 |
| SC      | 1085.7 |  2779.8 |  4854.7 |  1355.1 |
| SE      |   19.3 |    26.8 |    21.3 |    10.2 |
| SP      | 1739.2 | 28007   | 14096.3 | 20388.3 |
| TO      |  218.7 |    49.4 |    17.4 |   404.5 |

## Selected model per state (skill-gated)

**chikungunya**: loglin_trend: 8, ens_median: 7, xgb_direct: 6, ens_qavg: 2, seas_naive: 2, lgbm_direct: 1

**dengue**: loglin_trend: 12, lgbm_direct: 4, ens_median: 4, ens_qavg: 3, xgb_direct: 3

| state   | disease     | selected     |   mean_wis | fallback   |
|:--------|:------------|:-------------|-----------:|:-----------|
| AC      | chikungunya | ens_qavg     |        1.8 | False      |
| AC      | dengue      | loglin_trend |       48.9 | False      |
| AL      | chikungunya | loglin_trend |       39.4 | False      |
| AL      | dengue      | loglin_trend |       84.2 | False      |
| AM      | chikungunya | loglin_trend |        1.2 | False      |
| AM      | dengue      | loglin_trend |       39   | False      |
| AP      | chikungunya | lgbm_direct  |        1.9 | False      |
| AP      | dengue      | loglin_trend |       72.8 | False      |
| BA      | chikungunya | xgb_direct   |       88.2 | False      |
| BA      | dengue      | lgbm_direct  |     1282.8 | False      |
| CE      | chikungunya | xgb_direct   |      128.1 | False      |
| CE      | dengue      | ens_qavg     |      135.1 | False      |
| DF      | chikungunya | ens_median   |        3.9 | False      |
| DF      | dengue      | xgb_direct   |     1472.9 | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1380.6 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       70.3 | False      |
| MG      | chikungunya | ens_qavg     |     1121.6 | False      |
| MG      | dengue      | lgbm_direct  |     9744.1 | False      |
| MS      | chikungunya | ens_median   |       77   | False      |
| MS      | dengue      | ens_qavg     |      275.9 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | ens_qavg     |      198.4 | False      |
| PA      | chikungunya | xgb_direct   |        4.8 | False      |
| PA      | dengue      | loglin_trend |      107.5 | False      |
| PB      | chikungunya | xgb_direct   |       69.6 | False      |
| PB      | dengue      | loglin_trend |       98.9 | False      |
| PE      | chikungunya | ens_median   |       98.2 | False      |
| PE      | dengue      | loglin_trend |      144.8 | False      |
| PI      | chikungunya | ens_median   |       34.5 | False      |
| PI      | dengue      | ens_median   |      106.5 | False      |
| PR      | chikungunya | loglin_trend |       42.1 | False      |
| PR      | dengue      | ens_median   |     3253.2 | False      |
| RJ      | chikungunya | ens_median   |       23.5 | False      |
| RJ      | dengue      | xgb_direct   |     1800   | False      |
| RN      | chikungunya | ens_median   |       46.1 | False      |
| RN      | dengue      | loglin_trend |       87   | False      |
| RO      | chikungunya | loglin_trend |       26.7 | False      |
| RO      | dengue      | ens_median   |       56.8 | False      |
| RR      | chikungunya | ens_median   |        0.5 | False      |
| RR      | dengue      | lgbm_direct  |        4.3 | False      |
| RS      | chikungunya | loglin_trend |        3   | False      |
| RS      | dengue      | ens_median   |     1414.3 | False      |
| SC      | chikungunya | xgb_direct   |        4.7 | False      |
| SC      | dengue      | lgbm_direct  |     2240   | False      |
| SE      | chikungunya | seas_naive   |       13.9 | True       |
| SE      | dengue      | xgb_direct   |       17.5 | False      |
| SP      | chikungunya | loglin_trend |       84.4 | False      |
| SP      | dengue      | loglin_trend |    14020   | False      |
| TO      | chikungunya | xgb_direct   |       21.9 | False      |
| TO      | dengue      | loglin_trend |      146.6 | False      |

## Per-state hyperparameter tuning

104 tuned configurations (random search on a held-out 67-week window, WIS criterion; cached under `hyperparams/`). Mean gain over defaults: 18.6%; tuning improved WIS for 104/104 configs.

| state   | disease     | model       |   tuned_wis |   default_wis |   gain_pct |   max_depth |   n_estimators |
|:--------|:------------|:------------|------------:|--------------:|-----------:|------------:|---------------:|
| AC      | dengue      | xgb_direct  |       32.48 |         64.62 |       49.7 |           7 |            244 |
| AC      | dengue      | lgbm_direct |       31.88 |         58.97 |       45.9 |           7 |            244 |
| MA      | dengue      | xgb_direct  |       33.23 |         55.84 |       40.5 |           3 |            246 |
| MA      | dengue      | lgbm_direct |       33.35 |         54.61 |       38.9 |           3 |            246 |
| CE      | dengue      | xgb_direct  |      352.2  |        570.86 |       38.3 |           3 |            246 |
| CE      | dengue      | lgbm_direct |      360.02 |        558.12 |       35.5 |           3 |            246 |
| AM      | dengue      | lgbm_direct |       27.26 |         42.22 |       35.4 |           3 |            246 |
| SE      | dengue      | xgb_direct  |       26.51 |         40.5  |       34.6 |           3 |            246 |
| MT      | chikungunya | lgbm_direct |        1.22 |          1.81 |       32.6 |           5 |            390 |
| SE      | dengue      | lgbm_direct |       26.72 |         39.52 |       32.4 |           3 |            246 |
| MT      | chikungunya | xgb_direct  |        1.3  |          1.86 |       29.9 |           5 |            390 |
| SE      | chikungunya | xgb_direct  |       30.08 |         42.86 |       29.8 |           3 |            246 |
| RR      | chikungunya | lgbm_direct |        0.38 |          0.54 |       28.9 |           3 |            150 |
| AM      | dengue      | xgb_direct  |       27.45 |         38.56 |       28.8 |           3 |            246 |
| AL      | dengue      | lgbm_direct |      267.52 |        374.49 |       28.6 |           3 |            246 |
| MT      | dengue      | xgb_direct  |      240.82 |        336.37 |       28.4 |           3 |            246 |
| MS      | dengue      | lgbm_direct |      138.17 |        192.49 |       28.2 |           3 |            246 |
| RJ      | dengue      | lgbm_direct |       67.72 |         93.41 |       27.5 |           3 |            150 |
| PA      | dengue      | xgb_direct  |       36.61 |         50.33 |       27.3 |           3 |            246 |
| MA      | chikungunya | xgb_direct  |       18.91 |         25.95 |       27.1 |           3 |            246 |

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
