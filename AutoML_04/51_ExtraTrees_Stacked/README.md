# Summary of 51_ExtraTrees_Stacked

[<< Go back](../README.md)


## Extra Trees Regressor (Extra Trees)
- **n_jobs**: -1
- **criterion**: squared_error
- **max_features**: 0.7
- **min_samples_split**: 30
- **max_depth**: 7
- **eval_metric_name**: mae
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **shuffle**: True
 - **k_folds**: 10

## Optimized metric
mae

## Training time

46.2 seconds

### Metric details:
| Metric   |     Score |
|:---------|----------:|
| MAE      | 0.384492  |
| MSE      | 0.556867  |
| RMSE     | 0.746235  |
| R2       | 0.999859  |
| MAPE     | 0.0117358 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
