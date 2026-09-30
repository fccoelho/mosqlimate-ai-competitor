# IMDC Validation Report

Generated: 2026-09-30 10:37 · states: 26 · diseases: chikungunya, dengue

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
| loglin_trend | 1522.88 |             0.4  |          0.39 |
| ens_qavg     | 1628.33 |             0.36 |          0.35 |
| ens_median   | 1714.01 |             0.32 |          0.33 |
| lgbm_direct  | 1719.3  |             0.32 |          0.32 |
| xgb_direct   | 1748.81 |             0.31 |          0.32 |
| seas_naive   | 2527.72 |             0    |          0.38 |

### Best model per state (dengue)


loglin_trend: 20, ens_qavg: 5, ens_median: 1

### Mean WIS per state and test

| state   |      1 |       2 |       3 |       4 |
|:--------|-------:|--------:|--------:|--------:|
| AC      |   29.6 |    81   |    72.8 |   111.1 |
| AL      |  241.8 |   132.3 |    68.3 |    53.9 |
| AM      |   25.1 |    72.8 |    30.5 |    69.1 |
| AP      |   12.4 |   192.8 |    76.1 |    63.7 |
| BA      |  263.6 |  3410.7 |  1892.8 |   664.8 |
| CE      |  387.6 |    93.3 |    49.4 |   226.1 |
| DF      |  291.8 |  4052.7 |  2399.3 |   265.3 |
| GO      | 1019.6 |  3148.2 |  1844.5 |   812.4 |
| MA      |   28.9 |   106.8 |    48.3 |   143.7 |
| MG      | 4704.3 | 21520.1 | 15840.7 |  4247.7 |
| MS      |  416.5 |   465.4 |   122.1 |   127.8 |
| MT      |  133.8 |   154.1 |   180.5 |   512.3 |
| PA      |   23.1 |   240.1 |    92.9 |   166.1 |
| PB      |  344.2 |    85.2 |    60.6 |    63.6 |
| PE      |  229.4 |   313.7 |   100   |   102.6 |
| PI      |  228.6 |   115.8 |    58.2 |   164.9 |
| PR      | 1266.1 |  4739.7 |  6773.5 |  4102   |
| RJ      |  473.4 |  4751.9 |  2689.2 |   818.1 |
| RN      |  243.1 |   160.6 |    56.4 |    75   |
| RO      |   84.5 |   115.3 |    38.7 |    19.1 |
| RR      |    2.5 |     9.7 |     2.7 |     3.6 |
| RS      |  470.6 |  2535.3 |  1621.2 |  2845.8 |
| SC      | 1060.5 |  2604.6 |  5641   |  1268.2 |
| SE      |   20.2 |    28   |    28.4 |    12.8 |
| SP      | 1587.7 | 27829.6 | 14266.3 | 25938.5 |
| TO      |  222.3 |    41.5 |    17.1 |   495.9 |

## Selected model per state (skill-gated)

**chikungunya**: loglin_trend: 11, ens_median: 6, ens_qavg: 4, seas_naive: 3, lgbm_direct: 1, xgb_direct: 1

**dengue**: loglin_trend: 20, ens_qavg: 5, ens_median: 1

| state   | disease     | selected     |   mean_wis | fallback   |
|:--------|:------------|:-------------|-----------:|:-----------|
| AC      | chikungunya | loglin_trend |        1.9 | False      |
| AC      | dengue      | loglin_trend |       48.9 | False      |
| AL      | chikungunya | loglin_trend |       39.4 | False      |
| AL      | dengue      | loglin_trend |       84.2 | False      |
| AM      | chikungunya | ens_qavg     |        1.2 | False      |
| AM      | dengue      | loglin_trend |       39   | False      |
| AP      | chikungunya | lgbm_direct  |        1.8 | False      |
| AP      | dengue      | loglin_trend |       72.8 | False      |
| BA      | chikungunya | ens_median   |      108.2 | False      |
| BA      | dengue      | ens_qavg     |     1346   | False      |
| CE      | chikungunya | loglin_trend |      149.8 | False      |
| CE      | dengue      | loglin_trend |      137.2 | False      |
| DF      | chikungunya | ens_qavg     |        4.1 | False      |
| DF      | dengue      | ens_median   |     1457.7 | False      |
| GO      | chikungunya | loglin_trend |       68.6 | False      |
| GO      | dengue      | loglin_trend |     1380.6 | False      |
| MA      | chikungunya | seas_naive   |       11.6 | True       |
| MA      | dengue      | loglin_trend |       70.3 | False      |
| MG      | chikungunya | ens_qavg     |     1159.9 | False      |
| MG      | dengue      | loglin_trend |    10115.7 | False      |
| MS      | chikungunya | ens_qavg     |       76.5 | False      |
| MS      | dengue      | ens_qavg     |      256.2 | False      |
| MT      | chikungunya | loglin_trend |      316   | False      |
| MT      | dengue      | loglin_trend |      205.3 | False      |
| PA      | chikungunya | ens_median   |        4.9 | False      |
| PA      | dengue      | loglin_trend |      107.5 | False      |
| PB      | chikungunya | xgb_direct   |       83.6 | False      |
| PB      | dengue      | loglin_trend |       98.9 | False      |
| PE      | chikungunya | seas_naive   |      102.1 | True       |
| PE      | dengue      | loglin_trend |      144.8 | False      |
| PI      | chikungunya | loglin_trend |       36.4 | False      |
| PI      | dengue      | loglin_trend |      111.3 | False      |
| PR      | chikungunya | loglin_trend |       42.1 | False      |
| PR      | dengue      | loglin_trend |     3645.1 | False      |
| RJ      | chikungunya | ens_median   |       22.3 | False      |
| RJ      | dengue      | ens_qavg     |     1928.6 | False      |
| RN      | chikungunya | ens_median   |       46.5 | False      |
| RN      | dengue      | loglin_trend |       87   | False      |
| RO      | chikungunya | loglin_trend |       26.7 | False      |
| RO      | dengue      | ens_qavg     |       60.4 | False      |
| RR      | chikungunya | ens_median   |        0.5 | False      |
| RR      | dengue      | ens_qavg     |        4.4 | False      |
| RS      | chikungunya | loglin_trend |        3   | False      |
| RS      | dengue      | loglin_trend |     1445.7 | False      |
| SC      | chikungunya | loglin_trend |        5.1 | False      |
| SC      | dengue      | loglin_trend |     2381.2 | False      |
| SE      | chikungunya | seas_naive   |       13.9 | True       |
| SE      | dengue      | loglin_trend |       18.9 | False      |
| SP      | chikungunya | loglin_trend |       84.4 | False      |
| SP      | dengue      | loglin_trend |    14020   | False      |
| TO      | chikungunya | ens_median   |       23.9 | False      |
| TO      | dengue      | loglin_trend |      146.6 | False      |
