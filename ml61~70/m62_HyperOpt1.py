# pip install hyperopt
# 최소값을 찾는다.
# 베이지안옵티마이저는 최대값을 찾는다.
# print(hyperopt.__version__) # 0.2.7
import numpy as np
from hyperopt import  hp, fmin, tpe, Trials
import pandas as pd
search_space = {
    'x1' : hp.quniform('x1', -10, 10, 1),
    'x2' : hp.quniform('x2', -15, 15, 1)
    #      hp.quniform(label, low, high, q)
}
# print(search_space)
# {'x1': <hyperopt.pyll.base.Apply object at 0x0000018B662395E0>, 'x2': <hyperopt.pyll.base.Apply object at 0x0000018B6BC8CB80>}

def object_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1 ** 2 - 20 * x2
    return return_value

trial_val = Trials()

best = fmin(
    fn=object_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,   
    trials=trial_val,
    rstate= np.random.default_rng(seed=10)
)

# print('best: ', best)
# # best:  {'x1': 5.0, 'x2': 8.0}

# print(trial_val.results)
# #  {'loss': -24.0, 'status': 'ok'} ....
# print(trial_val.vals)

################### 데이터 프레임에 trial_val.vals를 넣어라!!! #####################

# trial_val = pd.DataFrame(trial_val)
# print(trial_val)

results = [aaa['loss'] for aaa in trial_val.results]
# 아래와 같다.
# for aaa in trial_val.results:
#     losses.append(aaa['loss'])

df = pd.DataFrame({'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2'],
                   'results' : results}
                   )

print(df)