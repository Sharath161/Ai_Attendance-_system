# Few-Shot Benchmark — real faces (LFW)

Protocol: 120 enrolled classes, fixed 5-photo test set per class, 114 unknown (open-set) probes, query uses flip-TTA. FAR target 1%.

| K shots | Strategy | Top-1 | Worst class | Class std | DIR@FAR1% |
|---|---|---|---|---|---|
| 1 | baseline | 97.3% | 0.0% | 10.6pp | 96.3% |
| 1 | +amplified | 97.5% | 0.0% | 10.5pp | 96.0% |
| 1 | +amplified +s-norm | 97.3% | 0.0% | 10.9pp | 97.0% |
| 2 | baseline | 98.5% | 80.0% | 5.3pp | 92.5% |
| 2 | +amplified | 98.5% | 80.0% | 5.3pp | 90.8% |
| 2 | +amplified +s-norm | 98.5% | 80.0% | 5.3pp | 97.7% |
| 3 | baseline | 98.2% | 60.0% | 6.3pp | 95.0% |
| 3 | +amplified | 98.3% | 80.0% | 5.5pp | 94.3% |
| 3 | +amplified +s-norm | 98.3% | 80.0% | 5.5pp | 97.8% |
| 5 | baseline | 98.3% | 60.0% | 6.1pp | 96.7% |
| 5 | +amplified | 98.5% | 80.0% | 5.3pp | 96.2% |
| 5 | +amplified +s-norm | 98.5% | 80.0% | 5.3pp | 98.2% |

## Speed (low-hardware profile)
| Profile | ms/image |
|---|---|
| full | 9.6 |
| edge(320) | 9.6 |

Figures: `fewshot_curves.png`, `fewshot_speed_profile.png`.