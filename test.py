import pandas as pd
from sklearn.metrics._regression import r2_score
from sklearn.preprocessing._polynomial import PolynomialFeatures
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
# import tensorflow as tf
# from keras import layers, models
# from keras.models import Sequential
# from keras.layers import Dense

# 모델에 사용할 칼럼들만 가져오도록 한다.
df=pd.read_csv("yongsan2021.csv")
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.
df['분기당_매출_금액']=df['분기당_매출_금액']/df['점포수']
gol= df[df['상권_구분_코드_명']=='골목상권']
# print(gol.info())
# print(gol.shape) #(2809, 9)
x=gol[['주중_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액','시간대_14~17_매출_금액','남성_매출_금액','연령대_40_매출_금액']].squeeze()
y=gol['분기당_매출_금액']
x=x.astype('float')
print(x.info())
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# x = poly_features.fit_transform(x)

#===============================================================================
# 이상치 제거
#===============================================================================
q1=gol['분기당_매출_금액'].quantile(0.25)
q2=gol['분기당_매출_금액'].quantile(0.5)
q3=gol['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
# print(iqr)

condition=gol['분기당_매출_금액']>q3+1.5*iqr
# print(data[condition])

a=gol[condition].index #480 개
gol.drop(a,inplace=True)
print(gol.shape) #(3750, 2)

# print(gol.describe())


# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(-1,1))
# x=scaler.fit_transform(x)












#
# from sklearn.linear_model import Ridge, LinearRegression
#
# ridge = Ridge(alpha=10).fit(train_x, train_y)
# print("훈련 세트 점수: {:.2f}".format(ridge.score(train_x, train_y)))
# print("테스트 세트 점수: {:.2f}".format(ridge.score(test_x, test_y)))
#
# lr = LinearRegression()
# model=lr.fit(train_x, train_y)
# print('y_new(예측값):\n', model.predict(test_x)[:5])
# print('실제값:\n', test_y.values[:5])
# print(r2_score(test_y,model.predict(test_x)))
# print("훈련 세트 점수: {:.2f}".format(lr.score(train_x, train_y)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(test_x, test_y)))

