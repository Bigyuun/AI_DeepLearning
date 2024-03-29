# 도미 데이터
bream_length = [25.4,	26.3,	26.5,	29,	29,	29.7,	29.7,	30,	30,	30.7,	31,	31,	31.5,	32,	32,	32,	33,	33,	33.5,	33.5,	34,	34,	34.5,	35,	35,	35,	35,	36,	36,	37,	38.5,	38.5,	39.5,	41,	41]
bream_weight = [242,	290,	340,	363,	430,	450,	500,	390,	450,	500,	475,	500,	500,	340,	600,	600,	700,	700,	610,	650,	575,	685,	620,	680,	700,	725,	720,	714,	850,	1000,	920,	955,	925,	975,	950]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 14.0, 14.3, 15.0]
smelt_weight = [6.7,7.5,7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# 도미+빙어 데이터
fish_length = bream_length+smelt_length
fish_weight = bream_weight+smelt_weight

# [길이,무게] 행렬로 전환
fish_data = [ [l,w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# 모니터링용
# print(fish_data)
# print(fish_target)
# print(len(fish_data))
# print(len(fish_target))
#===============================================================================

#===============================================================================
'''---------------------------------------------------------------------------
   기존 배열
   ---------------------------------------------------------------------------'''
   
# 모델 객체 생성
kn = KNeighborsClassifier()

# 도미데이터를 훈련데이터로 사용하기 위한 슬라이싱
train_input = fish_data[0:35]
train_target = fish_target[0:35]

# 빙어데이터를 테스트데이터로 사용하기 위한 슬라이싱
test_input =  fish_data[35:]
test_target = fish_target[35:]

'''---------------------------------------------------------------------------
   numpy 배열
   ---------------------------------------------------------------------------'''
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 도미와 빙어데이터를 섞어줌 --> index를 섞음
np.random.seed(42)   #seed는 랜덤함수를 일정하게 설정함(초기값 일정)
index = np.arange(49)
np.random.shuffle(index)

print(index)

train_input = input_arr[index[0:35]]
train_target = target_arr[index[0:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 단순 데이터 잘 들어가는지 확인용
print(train_input[0], input_arr[13])
#===============================================================================

#===============================================================================

# 그래프 도식화
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# Fit 모델 훈련
print("Fitting...")
kn.fit(train_input, train_target)
print("Fit complete")

score = kn.score(test_input, test_target)
print("Score : ", score)

print(kn.predict(test_input))

print(test_target)
