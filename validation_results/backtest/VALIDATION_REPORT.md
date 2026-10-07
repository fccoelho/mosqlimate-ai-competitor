# IMDC Validation Report

Generated: 2026-10-07 08:39 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts. `imdc_bb` is the 3rd IMDC procc
reference baseline (Mosqlimate model 85), scored locally on the
same weeks; skill_vs_imdc_bb > 0 means better than the baseline.

## Chikungunya — mean WIS per model (tests with actuals)

| model         |    wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:--------------|-------:|-----------------:|-------------------:|--------------:|
| timesfm_base  |  82.55 |             0.38 |               0.08 |          0.39 |
| timesfm       |  87.98 |             0.34 |               0.02 |          0.37 |
| imdc_bb       |  90.03 |             0.33 |               0    |          0.48 |
| ens_median_tf |  91.43 |             0.32 |              -0.02 |          0.35 |
| ens_qavg_tf   |  91.74 |             0.31 |              -0.02 |          0.38 |
| xgb_direct    |  95.31 |             0.29 |              -0.06 |          0.34 |
| ens_median    |  95.73 |             0.28 |              -0.06 |          0.34 |
| ens_qavg      |  96.33 |             0.28 |              -0.07 |          0.35 |
| lgbm_direct   | 100.19 |             0.25 |              -0.11 |          0.31 |
| ens_vote      | 104.04 |             0.22 |              -0.16 |          0.32 |
| loglin_trend  | 110.78 |             0.17 |              -0.23 |          0.36 |
| seas_naive    | 133.6  |             0    |              -0.48 |          0.29 |

### Best model per state (chikungunya)


imdc_bb: 10, timesfm_base: 7, timesfm: 3, seas_naive: 2, loglin_trend: 1, ens_qavg_tf: 1, ens_vote: 1, ens_median: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.8 |    3.3 |    1.4 |   1.7 |
| AL      |   58.8 |   52.4 |   44.5 |  26.5 |
| AM      |    1.6 |    1.4 |    1.2 |   0.8 |
| AP      |    0.5 |    4.9 |    1.4 |   0.7 |
| BA      |  104.8 |  139.9 |  126.4 |  98.3 |
| CE      |  371.9 |  147.1 |   25.5 |  26.7 |
| DF      |    3.8 |    5.1 |    2.9 |   3.6 |
| GO      |   22.8 |  127.8 |   71.9 | 155.3 |
| MA      |   15.6 |   29   |    8.5 |   6.7 |
| MG      | 1502.3 | 1263.6 | 1499.3 | 338.4 |
| MS      |   86.8 |   35.3 |  125.9 |  90.7 |
| MT      |    3.4 |  360   |  465.5 | 678.4 |
| PA      |    6.1 |    9.2 |    5   |   4.5 |
| PB      |  221.6 |   49.1 |   18   |   9.3 |
| PE      |  281.7 |   57.2 |   25.6 |  33.1 |
| PI      |   56.2 |   59.6 |   10.6 |   9.3 |
| PR      |   48.7 |   19.2 |   64.5 |  54   |
| RJ      |   39.1 |   48.4 |   34.4 |  36.2 |
| RN      |   94.7 |   52.7 |   28.8 |  24.5 |
| RO      |    0.9 |    2.8 |   81.3 |  32.7 |
| RR      |    0.6 |    0.7 |    0.7 |   0.5 |
| RS      |    1.5 |    2.9 |    4.3 |   4.4 |
| SC      |    4.1 |    3.4 |    8.3 |   5.7 |
| SE      |   31.9 |   31.3 |    9.8 |   5.8 |
| SP      |   36.8 |  108.8 |   81.9 | 121.7 |
| TO      |   42.5 |   41.4 |    6.4 |   4.6 |

## Dengue — mean WIS per model (tests with actuals)

| model         |     wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:--------------|--------:|-----------------:|-------------------:|--------------:|
| imdc_bb       | 1138.76 |             0.48 |               0    |          0.48 |
| ens_median_tf | 1373.41 |             0.38 |              -0.21 |          0.37 |
| ens_qavg      | 1389.4  |             0.37 |              -0.22 |          0.37 |
| ens_median    | 1389.87 |             0.37 |              -0.22 |          0.35 |
| xgb_direct    | 1392.64 |             0.37 |              -0.22 |          0.34 |
| loglin_trend  | 1410.84 |             0.36 |              -0.24 |          0.38 |
| ens_qavg_tf   | 1414.04 |             0.36 |              -0.24 |          0.38 |
| lgbm_direct   | 1426.46 |             0.35 |              -0.25 |          0.33 |
| timesfm_base  | 1473.68 |             0.33 |              -0.29 |          0.36 |
| ens_vote      | 1521.45 |             0.31 |              -0.34 |          0.34 |
| timesfm       | 1681.67 |             0.24 |              -0.48 |          0.4  |
| seas_naive    | 2209.05 |             0    |              -0.94 |          0.39 |

### Best model per state (dengue)


imdc_bb: 19, timesfm: 4, timesfm_base: 2, ens_qavg_tf: 1

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   29.9 |    72.1 |    68.7 |    59.2 |
| AL      |  172.5 |   126.4 |    52.8 |    38.4 |
| AM      |   25.7 |    61.6 |    23.8 |    49.1 |
| AP      |   12   |   190.5 |    57.5 |    43.4 |
| BA      |  271.6 |  3368   |  1141.6 |   409.1 |
| CE      |  264.7 |   102.3 |    62.4 |   138.7 |
| DF      |  251.7 |  4237.4 |  1576.5 |   184.1 |
| GO      |  870   |  3589.4 |  1299.1 |   582.6 |
| MA      |   26.1 |   104.3 |    37.3 |   122   |
| MG      | 4860.5 | 22719.1 | 11177.4 |  2196.3 |
| MS      |  370.6 |   350.1 |   149.5 |   141.8 |
| MT      |  130.5 |   179.7 |   222.1 |   246.3 |
| PA      |   22.8 |   244.6 |    86.3 |    81.9 |
| PB      |  202.7 |    91.7 |    55.8 |    42.9 |
| PE      |  197.8 |   242   |   147.6 |   108.4 |
| PI      |  172.3 |   106.1 |    38   |   101.8 |
| PR      | 1365.1 |  5202   |  5303.7 |  1979   |
| RJ      |  453.3 |  4696.2 |  1855   |   504.6 |
| RN      |  167   |   151.7 |    56.6 |    48.2 |
| RO      |   81.5 |    95.5 |    25   |    18.8 |
| RR      |    2.4 |     8.7 |     3.4 |     3.3 |
| RS      |  432.3 |  2642.2 |  1183.7 |  1200.6 |
| SC      | 1125   |  2583.2 |  3573.6 |  1510.3 |
| SE      |   18.3 |    24   |    18   |    11.5 |
| SP      | 1863.9 | 25682.6 | 12475.4 | 13175.5 |
| TO      |  167.5 |    45.4 |    25.7 |   262   |

## Selected model per state (skill-gated)

**chikungunya**: loglin_trend: 8, lgbm_direct: 5, ens_vote: 5, xgb_direct: 4, seas_naive: 2, ens_qavg: 1, ens_median: 1

**dengue**: loglin_trend: 11, lgbm_direct: 6, ens_qavg: 3, xgb_direct: 3, ens_vote: 2, ens_median: 1

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
| BA      | chikungunya | ens_vote     |       90.7 | False      |
| BA      | dengue      | lgbm_direct  |     1233.1 | False      |
| CE      | chikungunya | lgbm_direct  |      137.4 | False      |
| CE      | dengue      | ens_qavg     |      126.7 | False      |
| DF      | chikungunya | ens_vote     |        3.8 | False      |
| DF      | dengue      | xgb_direct   |     1420   | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1304.5 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       60.5 | False      |
| MG      | chikungunya | ens_qavg     |     1091.9 | False      |
| MG      | dengue      | lgbm_direct  |     9164   | False      |
| MS      | chikungunya | ens_vote     |       76.8 | False      |
| MS      | dengue      | ens_vote     |      242.1 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | xgb_direct   |      159.3 | False      |
| PA      | chikungunya | xgb_direct   |        3.8 | False      |
| PA      | dengue      | loglin_trend |      104.9 | False      |
| PB      | chikungunya | lgbm_direct  |       66.2 | False      |
| PB      | dengue      | loglin_trend |       88.8 | False      |
| PE      | chikungunya | xgb_direct   |       97.4 | False      |
| PE      | dengue      | ens_vote     |      153.9 | False      |
| PI      | chikungunya | ens_vote     |       34.6 | False      |
| PI      | dengue      | ens_qavg     |       93.3 | False      |
| PR      | chikungunya | loglin_trend |       42.1 | False      |
| PR      | dengue      | lgbm_direct  |     2963.9 | False      |
| RJ      | chikungunya | ens_vote     |       19.2 | False      |
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
