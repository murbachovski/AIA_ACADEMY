import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50])
# print(aaa.shape)
aaa = aaa.reshape(-1, 1)
# print(aaa.shape)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=.1) # 전체의 %를 이상치로 판단할 것이냐?
outlier.fit(aaa)
results = outlier.predict(aaa)
print(results)
