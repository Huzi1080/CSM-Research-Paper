# Adversarial Evaluation — Summary

## Aggregate Metrics

| dataset   |   accuracy |   precision |   recall |   f1_score |
|:----------|-----------:|------------:|---------:|-----------:|
| clean     |   0.22689  |    0.33894  | 0.376841 |   0.356887 |
| fgsm      |   0.326916 |    0.428849 | 0.549755 |   0.481833 |
| pgd       |   0.310726 |    0.416295 | 0.524351 |   0.464117 |

## Baseline Validation

| dataset   |   accuracy |   precision |   recall |   f1_score |
|:----------|-----------:|------------:|---------:|-----------:|
| baseline  |        0.9 |        0.89 |     0.91 |        0.9 |

## Figures

- accuracy_drop.png
- f1_drop.png
- performance_heatmap.png
- confusion_matrices
- roc_curves.png
- pr_curves.png
- risk_grid_map.png


*(Attack=positive class; AUC and AP computed from predicted probabilities.)*
