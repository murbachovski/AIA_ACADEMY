import numpy as np
from sklearn.covariance import EllipticEnvelope
ccc = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
            [100, 200, -30, 400, 500, 600, 700, -70000, 800, 900, 1000, 210, 420]]
            )
aaa = ccc.T
for i in range(aaa.shape[1]):
    column = aaa[:, i].reshape(-1, 1)
    # print(aaa[:, i].shape) # [-10   2   3   4   5   6 700   8   9  10  11  12  50] # (13,)
    # print(aaa[:, i].reshape(-1, 1).shape)
                                    # [[-10]
                                    #  [  2]
                                    #  [  3]
                                    #  [  4]
                                    #  [  5]
                                    #  [  6]
                                    #  [700]
                                    #  [  8]
                                    #  [  9]
                                    #  [ 10]
                                    #  [ 11]
                                    #  [ 12]
                                    #  [ 50]]
                                    # (13, 1)
    outlier = EllipticEnvelope(contamination=0.1)
    out = outlier.fit(column)
    results = outlier.predict(column)
    print(results)
