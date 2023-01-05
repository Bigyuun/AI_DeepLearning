# ==============================================================================
# 데이터 초기화
# ==============================================================================

# 도미 데이터
bream_length = [25.4,	26.3,	26.5,	29,	29,	29.7,	29.7,	30,	30,	30.7,	31,	31,	31.5,	32,	32,	32,	33,	33,	33.5,	33.5,	34,	34,	34.5,	35,	35,	35,	35,	36,	36,	37,	38.5,	38.5,	39.5,	41,	41]
bream_weight = [242,	290,	340,	363,	430,	450,	500,	390,	450,	500,	475,	500,	500,	340,	600,	600,	700,	700,	610,	650,	575,	685,	620,	680,	700,	725,	720,	714,	850,	1000,	920,	955,	925,	975,	950]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 14.0, 14.3, 15.0]
smelt_weight = [6.7,7.5,7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 전체 생선 데이터
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# 데이터 numpy 변환
# 튜플식으로 대입
fish_data = np.column_stack((fish_length, fish_weight)) # 2열의 구성
fish_target = np.concatenate( (np.ones(35), np.zeros(14)) ) # 리스트처럼 쭉

# ==============================================================================
# 라이브러리 Import
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 데이터 슬라이싱
# ==============================================================================

# train_test_split 함수 활용하기
# 한 배열을 두개로 쪼개줌 (인덱스는 섞음)
# 이 함수는 기본적으로 25%를 테스트 세트로 분리시킴
# fish_data --> 1. train_input, 2. test_input
# fish_target --> 1. train_target 2. test_target
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)  # random_state는 random.seed()와 같음

print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)

print(test_target)
''' test_target에는 도미 10, 빙어 3 개가 들어가있음. 즉 샘플링 편향이 일부 나타남 (초기에는 도미 35, 빙어 14인 2.5:1)'''
''' 따라서 이러한 편향을 바꿔줄 필요가 있음 ↓↓↓↓'''
# train_test_split 에 편향문제를 해결해줄 함수가 있음. 클래스의 비율에 맞춰 데이터를 나눠줌
# stratify를 타겟데이터로 설정해줌
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)

# ==============================================================================
# k - 최근접 이웃 사용
# ==============================================================================

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

kn.predict( [[25,150]]) 
''' 위 내용의 결과 '''
''' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'''
''' 값이 0.0 으로 나온다....'''
''' --> KNeighbors는 주어진 샘플에서 가장 가까운 이웃 5개를 기반으로 계산한다
    확인해보자'''

# 주변 이웃 다섯개의 거리와 인덱스를 추출
distances, index = kn.kneighbors([[25,150]])

# 그래프
plt.scatter(train_input[:,0], train_input[:,1],)
plt.scatter(25,150, marker='^')
plt.scatter(train_input[index,0], train_input[index,1], marker = 'D') # trian_input은 2x2 배열이므로 각 무게와 길이를 좌표점으로 찍음
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
''' 위 내용의 결과 '''
''' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'''
''' 알다시피 x, y axis의 scale이 다르다. 즉 distance 값이 실제로는 빙어 4개의 데이터와 가깝다는걸 알 수있다.
    그렇기에 함수는 빙어 4개를 채택한 것이다.'''

''' data preprocessing(데이터 전처리)작업이 필요하다'''
''' 가장 널리 알려진 방법인 표준점수 방법을 이용한다. (참고로 모든 알고리즘이 거리 기반인 것은 아니다)'''

mean = np.mean(train_input, axis=0) # 함수의 평균값, axis=0 : 행을 따라 각 열의 통계 값을 계산 (열하고싶으면 axis=1)
std = np.std(train_input, axis=0)   # 함수의 표준편차 (standard)

print(mean, std)

# 표준점수 = (원본-평균)/표준편차
# numpy의 브로드캐스팅기능으로 인해 전체 원본 배열에서 mean을 각각 빼주고 각각 std로 나눠준다.
train_scaled = (train_input-mean)/std

# ==============================================================================
# 전처리 된 데이터로 모델 훈련하기
# ==============================================================================

# 일단 모델 확인
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25,150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 아까 그 25, 150짜리 빙어를 전처리하고 대입
new_data = ([25,150]-mean)/std

# 전처리한 모델 재확인
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new_data[0], new_data[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

''' 위 내용의 결과 '''
''' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'''
''' 데이터의 산점도는 전과 거의 동일, 하지만 scale이 확연하게 좋아짐 (-1.5 ~ 1.5) '''

#  자... 다시 k-최근접 이웃 피팅
kn.fit(train_scaled, train_target)

# 테스트 데이터로 평가할 때도 스케일을 맞추기 위해 표준점수로 변환해줘야지~?
test_scaled = (test_input-mean)/std
kn.score(test_scaled, test_target)

print(kn.predict([new_data]))

''' scale이 바뀌었으니 distance도 도미에 가깝게 나올것이다~. '''
distance, index = kn.kneighbors([new_data])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new_data[0], new_data[1], marker='^')
plt.scatter(train_scaled[index,0], train_scaled[index,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
