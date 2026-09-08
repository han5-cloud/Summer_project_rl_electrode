# Ten-seed policy ablation and paired evaluation

All policies were evaluated on the same 1,000 scene seeds.
Fixed-sweep calibration threshold: 0.140

The hierarchical paired bootstrap resamples both the 10 training seeds and the shared scenes.

| Condition | Mean success | Sample SD | Mean false confirmation | Mean timeout |
|---|---:|---:|---:|---:|
| Combined | 94.16% | 1.84 pp | 1.01% | 4.83% |
| RGB-only | 94.87% | 1.55 pp | 3.99% | 1.14% |
| Relative-height-only | 93.61% | 1.66 pp | 1.97% | 4.42% |
