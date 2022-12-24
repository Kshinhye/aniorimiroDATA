import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font',family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

df=pd.read_csv("yongsan2021.csv")
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.
df['분기당_매출_금액']=df['분기당_매출_금액']/df['점포수']
gol= df[df['상권_구분_코드_명']=='골목상권']
gol= df[df['상권_구분_코드_명']=='골목상권']
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


x=gol[['시간대_06~11_매출_금액']]
y=gol[['분기당_매출_금액']]
print(x)
#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)

from sklearn.linear_model import LinearRegression
model=LinearRegression() #LinearRegression 
fit_model=model.fit(x,y)
print('기울기(slope, w):', fit_model.coef_) #회귀계수를 얻을 수 있다(w)
print('편향(bias, b):', fit_model.intercept_) #절편이라는 말 잘 안써(우리끼리 쓰는거얌), 편향이야 편향

#예측값 확인 함수로 미지의 feature에 대한 label을 예측
#print(xx[0]) #[-1.70073563]
# y_new=fit_model.predict(xx[0]) # err : 
# ValueError: Expected 2D array, got 1D array instead:
# array=[-1.70073563].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

#학습할떄 타입유지시키기 (sklearn에서는 matrix로 학습했기떄문에 matrix로 넣어준다.)
y_new=fit_model.predict(x[0:3])
print('y_new(예측값):', y_new) #[-52.17214291]
print('실제값:', y[0:3])  #-52.17214291

plt.scatter(x,y)
plt.show()