from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
# 수치화
token = Tokenizer()
token.fit_on_texts([text]) # 두 개 이상은 리스트.
# print(token.word_index)    # 많은 놈이 앞의 인덱스를 가진다. 그 후 앞에서 순서대로...
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])
# print(x) # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] => (1, 11) => 1행 11열
# print(type(x)) # <class 'list'>
# 가치가 높다고 판단. => 원핫 해줘야합니다.

######  1. to_categorical  ######
from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x) # 불필요한 0이 생긴다.
# # [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# print(x.shape)
# (1, 11, 9)

######  2. get_dummies  ######
import pandas as pd
x = pd.get_dummies(np.array(x).reshape(11, )) #넘파이로 바꿔준다 => # 1차원만 받습니다.
# x = pd.get_dummies(np.array(x).ravel()) # => 쫙 펴서 1차원으로 만든다. ravel()
# TypeError: unhashable type: 'list'
# 1. 넘파이로 바꿔준다. 2. 왜 리스트를 받지 못할까?
print(x)
