# September finalists on the newer data

The frozen RBF SVM ensemble at 1.25× persistence counts scored **0.414547** across the three mature October windows. The matched ensemble without SVM scored **0.414268**. The paired difference was **+0.000279**, with a 95% stable-user bootstrap interval of **[−0.004476, +0.004923]**. Full DFS at 1.0× and weighted PCA99 LR at 1.0× both scored higher than the RBF challenger.

## Fixed comparisons

Combined F1 pooled over 13,711 user-weeks and the three activities covered across all mature windows. Pooled F1 uses summed per-target confusion counts.

| Policy | Count multiplier | Combined F1 |
|---|---:|---:|
| H2 + neighbor-only DFS/PCA99 LR + EW7 | 1.10× | **0.419093** |
| H2 + F0 PCA99 LR + EW7 | 1.00× | 0.416745 |
| H2 + F0 PCA99 LR + standardized linear SVM + EW7 | 1.00× | 0.416493 |
| H2 + F0 PCA99 LR + PCA linear SVM + EW7 | 1.00× | 0.416131 |
| H2 + full DFS/PCA99 LR + EW7 | 1.00× | 0.415572 |
| H2 + F0 PCA99 LR + RBF SVM + EW7 | 1.00× | 0.415529 |
| H2/P021 | 1.00× | 0.415506 |
| H2 + F0 PCA99 LR + RBF SVM + EW7 | 1.25× | 0.414547 |
| H2 + F0 PCA99 LR + EW7 | 1.25× | 0.414268 |
| Standalone PCA linear SVM, earlier-calibrated margins | — | 0.394931 |
| Seven-day persistence | — | 0.358904 |
| Standalone PCA linear SVM, zero margin, C=1 | — | 0.308030 |

Neighbor-only DFS at its September-selected 1.1× count beat full DFS at 1.0× by +0.003521, with a 95% interval of [−0.001950, +0.009581]. That contrast changes both the DFS feature set and the count multiplier. At the same 1.1× setting, full DFS scored 0.419843.

## Threshold sensitivity

These are peaks of the already registered grids on these report weeks. The scores are descriptive because the peak was identified after scoring the newer data.

| Sweep | Highest pooled score | Setting |
|---|---:|---|
| Ensemble without SVM | 0.420573 | H2 + weighted PCA99 LR + EW7, 1.1× |
| Ensemble with SVM | 0.420551 | H2 + weighted PCA99 LR + PCA linear SVM + EW7, 1.1×, C=0.01 or 1 |
| RBF SVM ensemble | 0.417091 | 0.9× |
| Standalone SVM | 0.420523 | PCA linear SVM, margin ≥ −0.50, C=0.01 or 1 |

The standalone peak is within 0.00005 of the best no-SVM ensemble peak. The earlier-calibrated standalone procedure was lower by about 0.026, so the hindsight margin result needs separate confirmation. On the fixed comparisons, the RBF ensemble also scored lower at 1.25× than at 1.0×.

## Weekly coverage and the matched comparison

The three activities with hourly archive coverage throughout were opening a PR, opening an issue and submitting a PR review. The per-window covered scores also include issue comments and PR review comments on weeks where their archive coverage passes. Pushes were excluded every week. The November 1 Taiwan window is provisional and excluded from pooled metrics.

| Window start, Taiwan | User-weeks | Covered activities | RBF 1.25×, covered | No SVM 1.25×, covered | RBF 1.25×, common three | No SVM 1.25×, common three |
|---|---:|---:|---:|---:|---:|---:|
| Oct 11 | 4,790 | 5 | 0.407687 | 0.408720 | 0.419935 | 0.420837 |
| Oct 18 | 4,548 | 4 | 0.404474 | 0.408204 | 0.409715 | 0.407452 |
| Oct 25 | 4,373 | 3 | 0.412811 | 0.413186 | 0.412811 | 0.413186 |
| Nov 1, provisional | 4,202 | 3 | 0.443542 | 0.447448 | 0.443542 | 0.447448 |

## Reproduction

The replay used instructor revision `af9f335301d9badb482dfafa10c4534eabac845a` and the rebuilt archive through November 8 at 03:00 UTC, exclusive. The locked shortlist reused authenticated H2 predictions and refit the weighted PCA99 LR, DFS and selected SVM models at each origin on the permitted 56-day grid. The calibrated standalone SVM used an earlier 42-day fit, a seven-day purge and seven calibration origins. Threshold grids and the shortlist were registered before scoring. Historical H2 benchmark results were already available.

The verifier checked 536 weekly policy rows against the revised instructor evaluator and rebuilt decisions from the saved scores. It also verified the three activities' coverage from hourly receipts, matched the archived H2 and persistence results and replayed the registered hashes. The paired intervals resample 4,947 stable users with their weeks together, using 2,000 draws. They do not measure uncertainty across future weeks. The 45 focused PCA, DFS and SVM regression tests passed.

See `mature_common_target_metrics.csv` for the complete pooled grid, `weekly_metrics.csv` for weekly and six-target diagnostic scores, `paired_user_intervals.csv` for the fixed contrasts, and `hours_missing_by_activity.csv` for the coverage audit.
