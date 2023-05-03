from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

# 데이터 로드
X, y = load_diabetes(return_X_y=True)

# Bayesian Optimization을 위한 파라미터 범위 설정
pbounds = {'learning_rate': (0.01, 0.3),
           'max_depth': (2, 10),
           'subsample': (0.5, 1),
           'colsample_bytree': (0.5, 1),
           'gamma': (0, 5),
           'n_estimators': (50, 300)}

# 목적 함수 정의
def xgb_cv(learning_rate, max_depth, subsample, colsample_bytree, gamma, n_estimators):
    # XGBRegressor 모델 생성
    model = XGBRegressor(learning_rate=learning_rate,
                         max_depth=int(max_depth),
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         gamma=gamma,
                         n_estimators=int(n_estimators),
                         random_state=42)
    # 모델 교차 검증 수행
    score = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    return score

# Bayesian Optimization 객체 생성
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42,
)

# 최적의 하이퍼파라미터 탐색
optimizer.maximize(init_points=10, n_iter=20)

# 최적 하이퍼파라미터 추출
best_params = optimizer.max['params']

# 최적 하이퍼파라미터로 모델 생성 및 학습
model = XGBRegressor(**best_params, random_state=42)
model.fit(X, y)

# 예측 수행
y_pred = model.predict(X)

# R-squared 값 출력
print("R-squared:", r2_score(y, y_pred))
