# IMDC Validation Report

Generated: 2026-10-08 16:43 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts. `imdc_bb` is the 3rd IMDC procc
reference baseline (Mosqlimate model 85), scored locally on the
same weeks; skill_vs_imdc_bb > 0 means better than the baseline.

## Chikungunya — mean WIS per model (tests with actuals)

| model          |    wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:---------------|-------:|-----------------:|-------------------:|--------------:|
| timesfm_base   |  82.55 |             0.38 |               0.08 |          0.39 |
| timesfm_lagopt |  82.88 |             0.38 |               0.08 |          0.39 |
| timesfm        |  87.98 |             0.34 |               0.02 |          0.37 |
| imdc_bb        |  90.03 |             0.33 |               0    |          0.48 |
| ens_median_tf  |  91.43 |             0.32 |              -0.02 |          0.35 |
| ens_qavg_tf    |  91.74 |             0.31 |              -0.02 |          0.38 |
| xgb_direct     |  95.31 |             0.29 |              -0.06 |          0.34 |
| ens_median     |  95.73 |             0.28 |              -0.06 |          0.34 |
| ens_qavg       |  96.33 |             0.28 |              -0.07 |          0.35 |
| lgbm_direct    | 100.19 |             0.25 |              -0.11 |          0.31 |
| ens_vote       | 104.04 |             0.22 |              -0.16 |          0.32 |
| loglin_trend   | 110.78 |             0.17 |              -0.23 |          0.36 |
| xgb_lagopt     | 119.72 |             0.1  |              -0.33 |          0.31 |
| lgbm_lagopt    | 121.18 |             0.09 |              -0.35 |          0.3  |
| seas_naive     | 133.6  |             0    |              -0.48 |          0.29 |

### Best model per state (chikungunya)


imdc_bb: 9, timesfm_base: 5, timesfm_lagopt: 4, seas_naive: 2, timesfm: 2, loglin_trend: 1, ens_qavg_tf: 1, ens_vote: 1, ens_median: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.8 |    3.3 |    1.4 |   1.7 |
| AL      |   57.8 |   53.1 |   48.3 |  28.4 |
| AM      |    1.6 |    1.6 |    1.2 |   0.9 |
| AP      |    0.5 |    4.9 |    1.5 |   0.7 |
| BA      |  100   |  143.1 |  125.7 | 100.8 |
| CE      |  354.9 |  137.2 |   22.6 |  24.5 |
| DF      |    3.7 |    5   |    2.7 |   3.9 |
| GO      |   25.3 |  131.1 |   76.7 | 158.2 |
| MA      |   15.4 |   30.1 |    7.8 |   6.4 |
| MG      | 1523.4 | 1369.9 | 1593.9 | 297.8 |
| MS      |   86.8 |   33.4 |  124.3 |  86.3 |
| MT      |    3.2 |  362.5 |  469.2 | 701.5 |
| PA      |    5.6 |    8.7 |    5.3 |   4.4 |
| PB      |  235   |   46.2 |   15.6 |   7.8 |
| PE      |  291.7 |   53.6 |   24.9 |  33.4 |
| PI      |   54.6 |   62.9 |   10.8 |   9.1 |
| PR      |   48.8 |   19.8 |   65   |  57.3 |
| RJ      |   43.6 |   46.1 |   32.7 |  35.4 |
| RN      |   89.7 |   52.4 |   29.2 |  23.9 |
| RO      |    0.9 |    2.9 |   81.1 |  33.1 |
| RR      |    0.6 |    0.8 |    0.7 |   0.5 |
| RS      |    1.5 |    2.8 |    4.3 |   5.1 |
| SC      |    4   |    3.5 |    8.2 |   6.1 |
| SE      |   32.3 |   29.6 |    8.4 |   5.8 |
| SP      |   36.1 |  116.9 |   84.4 | 114.9 |
| TO      |   42   |   39.4 |    5.8 |   4.3 |

## Dengue — mean WIS per model (tests with actuals)

| model          |     wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:---------------|--------:|-----------------:|-------------------:|--------------:|
| imdc_bb        | 1138.76 |             0.48 |               0    |          0.48 |
| ens_median_tf  | 1373.41 |             0.38 |              -0.21 |          0.37 |
| ens_qavg       | 1389.4  |             0.37 |              -0.22 |          0.37 |
| ens_median     | 1389.87 |             0.37 |              -0.22 |          0.35 |
| xgb_direct     | 1392.64 |             0.37 |              -0.22 |          0.34 |
| loglin_trend   | 1410.84 |             0.36 |              -0.24 |          0.38 |
| ens_qavg_tf    | 1414.04 |             0.36 |              -0.24 |          0.38 |
| lgbm_direct    | 1426.46 |             0.35 |              -0.25 |          0.33 |
| timesfm_base   | 1473.68 |             0.33 |              -0.29 |          0.36 |
| ens_vote       | 1521.45 |             0.31 |              -0.34 |          0.34 |
| xgb_lagopt     | 1617.75 |             0.27 |              -0.42 |          0.36 |
| timesfm        | 1681.67 |             0.24 |              -0.48 |          0.4  |
| lgbm_lagopt    | 1696.66 |             0.23 |              -0.49 |          0.35 |
| timesfm_lagopt | 1718.51 |             0.22 |              -0.51 |          0.38 |
| seas_naive     | 2209.05 |             0    |              -0.94 |          0.39 |

### Best model per state (dengue)


imdc_bb: 18, timesfm: 4, timesfm_base: 2, timesfm_lagopt: 1, ens_qavg_tf: 1

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   28.1 |    77.6 |    65.6 |    68.7 |
| AL      |  180.5 |   125.9 |    52.2 |    37.3 |
| AM      |   25.7 |    61.2 |    26.3 |    46.2 |
| AP      |   11.7 |   190.5 |    54   |    43.9 |
| BA      |  290.1 |  3350.9 |  1365.4 |   380.8 |
| CE      |  302.4 |   104.1 |    56.9 |   148.2 |
| DF      |  246.4 |  4251.3 |  1511.2 |   177.3 |
| GO      |  989.1 |  3639.4 |  1302.9 |   594.5 |
| MA      |   26.8 |   107.3 |    36.6 |   118.7 |
| MG      | 4825   | 22626.7 | 12040.2 |  2076.6 |
| MS      |  397.4 |   365.2 |   164.4 |   133.7 |
| MT      |  127   |   169.2 |   218.6 |   256.8 |
| PA      |   24.4 |   245.4 |    87.6 |    84.5 |
| PB      |  215.6 |    92.8 |    58.1 |    41.1 |
| PE      |  198.1 |   234   |   142.3 |   106.5 |
| PI      |  188.3 |   106.1 |    38.7 |   108.4 |
| PR      | 1351.1 |  5307.7 |  5262   |  1850.3 |
| RJ      |  433.5 |  4691.2 |  2064.3 |   492.8 |
| RN      |  170   |   150.4 |    61.5 |    47.8 |
| RO      |   80.7 |   105.8 |    27.3 |    21.5 |
| RR      |    2.4 |     8.7 |     3.5 |     3.4 |
| RS      |  441.3 |  2625.4 |  1146.3 |  1309.1 |
| SC      | 1060.1 |  2538.5 |  3870.5 |  1826.9 |
| SE      |   19   |    24.6 |    17.1 |    12.6 |
| SP      | 1888.1 | 25451.5 | 14057.5 | 14048.5 |
| TO      |  182   |    44.2 |    28.4 |   259.6 |

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

## Covariate lag selection

270 combinations: climate->cases lags estimated per
state/disease/test (Spearman cross-correlation of seasonally
adjusted series, 0-16 weeks), then stepwise inclusion under
rolling-origin CV. Median lag of *selected* covariates: cf_precip_tot=9wk, cf_temp_med=10wk, cf_umid_med=8wk, oc_enso=5wk, oc_iod=14wk, oc_pdo=4wk

| state   | disease     | test   | selected                                |
|:--------|:------------|:-------|:----------------------------------------|
| AC      | chikungunya | 1      | cf_umid_med, cf_precip_tot, cf_temp_med |
| AC      | chikungunya | 2      | cf_temp_med, oc_pdo                     |
| AC      | chikungunya | 3      | cf_temp_med, oc_enso                    |
| AC      | chikungunya | 4      | -                                       |
| AC      | chikungunya | final  | -                                       |
| AL      | chikungunya | 1      | cf_temp_med, cf_umid_med, cf_precip_tot |
| AL      | chikungunya | 2      | -                                       |
| AL      | chikungunya | 3      | oc_enso                                 |
| AL      | chikungunya | 4      | cf_temp_med                             |
| AL      | chikungunya | final  | cf_umid_med, oc_pdo, cf_temp_med        |
| AM      | chikungunya | 1      | oc_enso                                 |
| AM      | chikungunya | 2      | cf_precip_tot, cf_temp_med, cf_umid_med |
| AM      | chikungunya | 3      | oc_iod, cf_umid_med                     |
| AM      | chikungunya | 4      | -                                       |
| AM      | chikungunya | final  | oc_enso                                 |
| AP      | chikungunya | 1      | -                                       |
| AP      | chikungunya | 2      | -                                       |
| AP      | chikungunya | 3      | -                                       |
| AP      | chikungunya | 4      | oc_pdo, oc_enso                         |
| AP      | chikungunya | final  | oc_pdo                                  |

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
