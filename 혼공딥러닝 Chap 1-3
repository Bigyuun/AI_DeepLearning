# 혼공 딥러닝 셀프스터디 자료
# 교재에 따라 Colab 환경 진행

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

bream_length = [25.4,	26.3,	26.5,	29,	29,	29.7,	29.7,	30,	30,	30.7,	31,	31,	31.5,	32,	32,	32,	33,	33,	33.5,	33.5,	34,	34,	34.5,	35,	35,	35,	35,	36,	36,	37,	38.5,	38.5,	39.5,	41,	41]
bream_weight = [242,	290,	340,	363,	430,	450,	500,	390,	450,	500,	475,	500,	500,	340,	600,	600,	700,	700,	610,	650,	575,	685,	620,	680,	700,	725,	720,	714,	850,	1000,	920,	955,	925,	975,	950]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 14.0, 14.3, 15.0]
smelt_weight = [6.7,7.5,7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = bream_length+smelt_length
weight = bream_weight+smelt_weight

fish_data = [ [l,w] for l, w in zip(length, weight)]
fish_target = [1]*35 + [0]*14

print(fish_data)
print(fish_target)
print(len(fish_data))
print(len(fish_target))

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)

kn.predict([[30,600]])
