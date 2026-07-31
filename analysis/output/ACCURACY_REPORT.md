# Accuracy Enhancement — Recognition Model

Protocol: Olivetti 40 identities, enrol 7 / test 3. Global threshold 0.35; recalibrated threshold from DET at FMR=1%.


## ArcFace-MobileNet (w600k_mbf)

| Strategy | Top-1 | EER | TAR@FMR1% | Best-F1 | Best-t | F1@0.35 |
|---|---|---|---|---|---|---|
| S0 baseline (mean proto) | 72.5% | 14.3% | 56.7% | 0.597 | 0.83 | 0.049 |
| S1 +flip TTA | 81.7% | 11.4% | 59.2% | 0.629 | 0.86 | 0.049 |
| S2 +quality-weighted proto | 81.7% | 11.4% | 60.0% | 0.629 | 0.86 | 0.049 |
| S3 multi-prototype (max-sim) | 80.8% | 12.2% | 71.7% | 0.746 | 0.86 | 0.049 |

## ArcFace-ResNet50 (w600k_r50)

| Strategy | Top-1 | EER | TAR@FMR1% | Best-F1 | Best-t | F1@0.35 |
|---|---|---|---|---|---|---|
| S0 baseline (mean proto) | 61.7% | 23.0% | 36.7% | 0.438 | 0.90 | 0.049 |
| S1 +flip TTA | 72.5% | 15.8% | 37.5% | 0.444 | 0.92 | 0.049 |
| S2 +quality-weighted proto | 72.5% | 16.2% | 37.5% | 0.444 | 0.92 | 0.049 |
| S3 multi-prototype (max-sim) | 75.8% | 16.4% | 44.2% | 0.498 | 0.92 | 0.049 |

**Best strategy:** S3 multi-prototype (max-sim). 
Figures: `acc_det_curve.png`, `acc_scores_hist.png`, `acc_strategy_bars.png`.