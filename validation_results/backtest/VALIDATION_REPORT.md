# IMDC Validation Report

Generated: 2026-10-08 17:22 · states: 26 · diseases: chikungunya, dengue

Metric: Weighted Interval Score (Bracher et al. 2021), lower is better,
computed weekly over the 52-week target season on conformally
calibrated quantile forecasts. `imdc_bb` is the 3rd IMDC procc
reference baseline (Mosqlimate model 85), scored locally on the
same weeks; skill_vs_imdc_bb > 0 means better than the baseline.

## Chikungunya — mean WIS per model (tests with actuals)

| model            |    wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:-----------------|-------:|-----------------:|-------------------:|--------------:|
| timesfm_base     |  82.55 |             0.38 |               0.08 |          0.39 |
| timesfm_lagopt   |  82.88 |             0.38 |               0.08 |          0.39 |
| ens_blend_recal  |  85.23 |             0.36 |               0.05 |          0.47 |
| ens_blend        |  86.44 |             0.35 |               0.04 |          0.39 |
| timesfm          |  87.98 |             0.34 |               0.02 |          0.37 |
| imdc_bb          |  90.03 |             0.33 |               0    |          0.48 |
| ens_median_tf    |  91.43 |             0.32 |              -0.02 |          0.35 |
| ens_qavg_tf      |  91.74 |             0.31 |              -0.02 |          0.38 |
| ens_median_recal |  93.55 |             0.3  |              -0.04 |          0.4  |
| ens_qavg_recal   |  95.2  |             0.29 |              -0.06 |          0.42 |
| xgb_direct       |  95.31 |             0.29 |              -0.06 |          0.34 |
| ens_median       |  95.73 |             0.28 |              -0.06 |          0.34 |
| ens_qavg         |  96.33 |             0.28 |              -0.07 |          0.35 |
| lgbm_direct      | 100.19 |             0.25 |              -0.11 |          0.31 |
| ens_vote         | 104.04 |             0.22 |              -0.16 |          0.32 |
| loglin_trend     | 110.78 |             0.17 |              -0.23 |          0.36 |
| xgb_lagopt       | 119.72 |             0.1  |              -0.33 |          0.31 |
| lgbm_lagopt      | 121.18 |             0.09 |              -0.35 |          0.3  |
| seas_naive       | 133.6  |             0    |              -0.48 |          0.29 |

### Best model per state (chikungunya)


imdc_bb: 9, timesfm_base: 5, timesfm_lagopt: 3, seas_naive: 2, ens_blend_recal: 1, loglin_trend: 1, ens_qavg_recal: 1, ens_qavg_tf: 1, ens_median_recal: 1, timesfm: 1, ens_median: 1

### Mean WIS per state and test

| state   |      1 |      2 |      3 |     4 |
|:--------|-------:|-------:|-------:|------:|
| AC      |    0.7 |    3.2 |    1.4 |   1.7 |
| AL      |   54.1 |   53.1 |   45.1 |  27.4 |
| AM      |    1.6 |    1.5 |    1.2 |   0.8 |
| AP      |    0.5 |    4.9 |    1.4 |   0.7 |
| BA      |   97.3 |  137.7 |  120   |  95.6 |
| CE      |  353.3 |  139.4 |   21.2 |  24.4 |
| DF      |    3.7 |    5.1 |    2.7 |   3.8 |
| GO      |   24.3 |  127.7 |   74.1 | 154.3 |
| MA      |   15.4 |   29.4 |    7.8 |   6.1 |
| MG      | 1522.6 | 1323.7 | 1543.5 | 295.6 |
| MS      |   86.4 |   33.4 |  120.4 |  84.7 |
| MT      |    3   |  363.7 |  464.3 | 675.9 |
| PA      |    5.1 |    8.2 |    5.1 |   4   |
| PB      |  224.5 |   45.8 |   15.8 |   7.6 |
| PE      |  286.5 |   53.8 |   24.2 |  32.4 |
| PI      |   54.3 |   62.8 |   11.2 |   9.5 |
| PR      |   48.6 |   19.5 |   61.4 |  54.8 |
| RJ      |   42.4 |   44.1 |   30.8 |  33.9 |
| RN      |   87.5 |   52.5 |   28.9 |  24.4 |
| RO      |    0.9 |    2.9 |   80.9 |  32   |
| RR      |    0.6 |    0.8 |    0.6 |   0.5 |
| RS      |    1.5 |    2.7 |    4.2 |   4.9 |
| SC      |    4   |    3.5 |    8.1 |   5.8 |
| SE      |   31.2 |   29.5 |    8.8 |   5.6 |
| SP      |   35.8 |  111.9 |   82   | 114.4 |
| TO      |   41.1 |   38.9 |    5.7 |   4.3 |

## Dengue — mean WIS per model (tests with actuals)

| model          |     wis |   skill_vs_naive |   skill_vs_imdc_bb |   coverage_50 |
|:---------------|--------:|-----------------:|-------------------:|--------------:|
| imdc_bb        | 1138.76 |             0.48 |               0    |          0.48 |
| ens_median_tf  | 1373.41 |             0.38 |              -0.21 |          0.37 |
| ens_qavg       | 1389.4  |             0.37 |              -0.22 |          0.37 |
| ens_median     | 1389.87 |             0.37 |              -0.22 |          0.35 |
| xgb_direct     | 1392.64 |             0.37 |              -0.22 |          0.34 |
| ens_blend      | 1396.01 |             0.37 |              -0.23 |          0.37 |
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
| AC      |   28.7 |    75.5 |    64.9 |    68.3 |
| AL      |  176   |   125.5 |    51.7 |    36.7 |
| AM      |   25.5 |    61   |    26   |    47.5 |
| AP      |   11.8 |   190.4 |    53.3 |    44   |
| BA      |  290.5 |  3357.8 |  1329.1 |   383.5 |
| CE      |  298.6 |   105.1 |    55.1 |   147.2 |
| DF      |  244.5 |  4262   |  1488.1 |   177.2 |
| GO      |  961.3 |  3657   |  1289.6 |   588.9 |
| MA      |   26.4 |   107.8 |    36.4 |   118   |
| MG      | 4845.9 | 22740.9 | 11759.7 |  2066.1 |
| MS      |  398.2 |   357   |   166.7 |   139.6 |
| MT      |  126.4 |   171   |   215.8 |   257.9 |
| PA      |   24.3 |   246.6 |    87.3 |    83.5 |
| PB      |  209.1 |    92.2 |    58.6 |    40.8 |
| PE      |  197.3 |   230.4 |   142.1 |   104.8 |
| PI      |  184.4 |   105.9 |    38.5 |   107.6 |
| PR      | 1360.7 |  5332.6 |  5349.6 |  1827.7 |
| RJ      |  433.7 |  4701.7 |  2023.5 |   489.8 |
| RN      |  164   |   149.4 |    61.7 |    47.2 |
| RO      |   80.7 |   104.3 |    27   |    21.4 |
| RR      |    2.4 |     8.7 |     3.5 |     3.3 |
| RS      |  438   |  2619.8 |  1146.9 |  1322.6 |
| SC      | 1071.2 |  2522.1 |  3917.6 |  1997.7 |
| SE      |   18.6 |    24.3 |    17.4 |    12.4 |
| SP      | 1857.8 | 25205.3 | 13631.5 | 13908.8 |
| TO      |  177.5 |    44.2 |    29.8 |   260.5 |

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
