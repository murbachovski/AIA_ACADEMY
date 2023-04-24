# Summary of 61_CatBoost_BoostOnErrors

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.05
- **depth**: 6
- **rsm**: 1
- **loss_function**: RMSE
- **eval_metric**: MAE
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **shuffle**: True
 - **k_folds**: 10

## Optimized metric
mae

## Training time

134.0 seconds

### Metric details:
| Metric   |     Score |
|:---------|----------:|
| MAE      | 0.434423  |
| MSE      | 0.515503  |
| RMSE     | 0.717985  |
| R2       | 0.999869  |
| MAPE     | 0.0119943 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
