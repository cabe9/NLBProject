# Validation Residual Diagnostic

Reference prediction set: `current_pair`.
Positive `delta_vs_reference` means lower Poisson residual loss than the reference.

## Overall

| model | slice | mean_loss | delta_vs_reference | target_spikes |
| --- | --- | --- | --- | --- |
| screen_k_width | overall | 0.066958 | 0.000018 | 53448.000000 |
| screen_l_spike125 | overall | 0.066966 | 0.000010 | 53448.000000 |
| screen_m_spike125 | overall | 0.066967 | 0.000009 | 53448.000000 |
| current_pair | overall | 0.066976 | 0.000000 | 53448.000000 |

## Best Unit-Level Improvements

| model | unit | mean_loss | delta_vs_reference | target_spikes |
| --- | --- | --- | --- | --- |
| screen_k_width | 1 | 0.041643 | 0.000099 | 646.000000 |
| screen_k_width | 17 | 0.114038 | 0.000077 | 2102.000000 |
| screen_k_width | 39 | 0.065524 | 0.000075 | 1047.000000 |
| screen_k_width | 43 | 0.186525 | 0.000073 | 3834.000000 |
| screen_k_width | 29 | 0.248336 | 0.000072 | 5641.000000 |
| screen_k_width | 27 | 0.046604 | 0.000063 | 690.000000 |
| screen_l_spike125 | 35 | 0.077008 | 0.000061 | 1451.000000 |
| screen_m_spike125 | 2 | 0.082894 | 0.000061 | 1394.000000 |
| screen_k_width | 32 | 0.059787 | 0.000059 | 1131.000000 |
| screen_l_spike125 | 39 | 0.065542 | 0.000058 | 1047.000000 |
| screen_k_width | 2 | 0.082899 | 0.000056 | 1394.000000 |
| screen_l_spike125 | 2 | 0.082899 | 0.000056 | 1394.000000 |

## Largest Unit-Level Regressions

| model | unit | mean_loss | delta_vs_reference | target_spikes |
| --- | --- | --- | --- | --- |
| screen_k_width | 23 | 0.018532 | -0.000043 | 235.000000 |
| screen_m_spike125 | 32 | 0.059876 | -0.000030 | 1131.000000 |
| screen_k_width | 9 | 0.089501 | -0.000029 | 1598.000000 |
| screen_l_spike125 | 32 | 0.059872 | -0.000027 | 1131.000000 |
| screen_l_spike125 | 34 | 0.046603 | -0.000025 | 745.000000 |
| screen_m_spike125 | 31 | 0.020701 | -0.000024 | 292.000000 |
| screen_l_spike125 | 31 | 0.020698 | -0.000021 | 292.000000 |
| screen_l_spike125 | 13 | 0.047101 | -0.000020 | 733.000000 |
| screen_m_spike125 | 29 | 0.248427 | -0.000019 | 5641.000000 |
| screen_k_width | 11 | 0.081670 | -0.000017 | 1311.000000 |
| screen_l_spike125 | 41 | 0.021654 | -0.000016 | 299.000000 |
| screen_m_spike125 | 41 | 0.021653 | -0.000015 | 299.000000 |

## Best Condition-Level Improvements

| field | value | model | n_trials | mean_loss | delta_vs_reference | target_spikes |
| --- | --- | --- | --- | --- | --- | --- |
| trial_type | 7.000000 | screen_k_width | 16 | 0.070175 | 0.000070 | 1553.000000 |
| maze_id | 7.000000 | screen_k_width | 16 | 0.070175 | 0.000070 | 1553.000000 |
| trial_type | 32.000000 | screen_k_width | 16 | 0.070032 | 0.000062 | 1610.000000 |
| maze_id | 88.000000 | screen_k_width | 16 | 0.070032 | 0.000062 | 1610.000000 |
| maze_id | 2.000000 | screen_k_width | 15 | 0.065707 | 0.000057 | 1373.000000 |
| trial_type | 2.000000 | screen_k_width | 15 | 0.065707 | 0.000057 | 1373.000000 |
| maze_id | 96.000000 | screen_k_width | 17 | 0.069741 | 0.000055 | 1670.000000 |
| trial_type | 38.000000 | screen_k_width | 17 | 0.069741 | 0.000055 | 1670.000000 |
| maze_id | 7.000000 | screen_m_spike125 | 16 | 0.070195 | 0.000050 | 1553.000000 |
| trial_type | 7.000000 | screen_m_spike125 | 16 | 0.070195 | 0.000050 | 1553.000000 |
| trial_type | 2.000000 | screen_m_spike125 | 15 | 0.065714 | 0.000050 | 1373.000000 |
| maze_id | 2.000000 | screen_m_spike125 | 15 | 0.065714 | 0.000050 | 1373.000000 |
