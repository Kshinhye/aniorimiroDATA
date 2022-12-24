import pandas as pd
from sklearn.metrics._regression import r2_score
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font',family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf

#temp와 atemp는 상관관계가 높은 것 같다. 긍분산성, 다중분산성 문제가 발생할 가능성이 높다. 둘 중 하나만 사용하는것도 고려
# 모델에 사용할 칼럼들만 가져오도록 한다.
df=pd.read_csv("yongsan2021.csv")
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.
df['분기당_매출_금액']=df['분기당_매출_금액']/df['점포수']
jdf= df[df['상권_구분_코드_명']=='전통시장']

#===============================================================================
# 이상치 제거
#===============================================================================
q1=jdf['분기당_매출_금액'].quantile(0.25)
q2=jdf['분기당_매출_금액'].quantile(0.5)
q3=jdf['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
# print(jdf)

condition=jdf['분기당_매출_금액']>q3+1.5*iqr
# print(data[condition])

a=jdf[condition].index
jdf.drop(a,inplace=True)


print(jdf.info()) # 5  16  22  7  21  12
print(jdf.shape) #(528, 47) -> (476, 47)
 
x=jdf[['여성_매출_금액','연령대_60_이상_매출_건수','주중_매출_금액','연령대_50_매출_금액','시간대_14~17_매출_금액']]
y=jdf['분기당_매출_금액']

#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)
#합습할 때 fit()의 파라미터값이 중요한것은 딥러닝이다.
import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=jdf).fit()
print(lm.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.396
# Model:                            OLS   Adj. R-squared:                  0.390
# Method:                 Least Squares   F-statistic:                     61.74
# Date:                Sat, 24 Dec 2022   Prob (F-statistic):           2.04e-49
# Time:                        01:53:16   Log-Likelihood:                -8639.7
# No. Observations:                 476   AIC:                         1.729e+04
# Df Residuals:                     470   BIC:                         1.732e+04
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   1.354e+07   9.91e+05     13.658      0.000    1.16e+07    1.55e+07
# x[0]           0.0668      0.023      2.893      0.004       0.021       0.112
# x[1]          -0.0192      0.018     -1.086      0.278      -0.054       0.016
# x[2]           0.0370      0.008      4.657      0.000       0.021       0.053
# x[3]          -0.1488      0.031     -4.759      0.000      -0.210      -0.087
# x[4]           0.0467      0.024      1.909      0.057      -0.001       0.095
# ==============================================================================
# Omnibus:                       98.987   Durbin-Watson:                   1.909
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              229.661
# Skew:                           1.067   Prob(JB):                     1.35e-50
# Kurtosis:                       5.650   Cond. No.                     5.56e+08
# ==============================================================================
print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np
df_lm=jdf.iloc[:,[5,16,22,7,21,12]]

fitted=lm.predict(df_lm) #이얏 예측값을 얻겠지
print('예측값: ',fitted.values[:3])
print('실제값: ', y.values[:3])
residual=df_lm['분기당_매출_금액']-fitted #잔차


# print(residual.head(3))
# print('잔차의 평균:', np.mean(residual))  #잔차의 평균: 1.249067923601936e-08
#
# #===============================================================================
# print('---선형성---') # 불만족
# #===============================================================================
# sns.regplot(fitted,residual,lowess=True, line_kws={'color':'red'})
# plt.plot([fitted.min(),fitted.max()],[0,0],'--',color='blue')
# plt.show()
# #잔차가 일정하게 분포되어있으므로 선형성 만족
# #===============================================================================
# print('---정규성---') # 불만족
# #===============================================================================
# import scipy.stats as stats
# sr=stats.zscore(residual)
# (x,y),_=stats.probplot(sr)
# sns.scatterplot(x,y)
# plt.plot([-3,3],[-3,3],'--',color='yellow')
# plt.show()
# #찰-싹! 붙어있죠. 잔차항이 정규분포를 따름
# #shapiro도 볼 수 있다. 0.05보다 커야해요
# print('shapito test: ',stats.shapiro(residual))
# # ShapiroResult(statistic=0.49560707807540894, pvalue=4.254312004167968e-36)
# #pvalue=0.0 얘만 관심있다. < 0.05 정규성불만족
#
# #===============================================================================
# print('---독립성---')
# #===============================================================================
# #Durbin-Watson: 1.909
#
# #===============================================================================
# print('---등분산성---') # 불만족
# #===============================================================================
# #오차들의 분산은 일정해야해
# sr=stats.zscore(residual)
# sns.regplot(fitted,np.sqrt(abs(sr)),lowess=True, line_kws={'color':'red'})
# plt.show()
# #평펴엉~합니다 등분산성 만족이에요
# #평균선을 기준으로 일정한 패턴을 보이지 않아 등분산성 만족이야
#
# #===============================================================================
# print('---다중공선성---')
# #===============================================================================
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# df2=jdf[['여성_매출_금액','연령대_60_이상_매출_금액','주중_매출_금액','연령대_50_매출_금액','시간대_14~17_매출_금액']]
# # print(df2.head(2))
# # print(df2.shape) #(2809, 6)
# #분산팽창계수를 사용하도록 할게요
# vifdf=pd.DataFrame()
# vifdf['vif_value']=[variance_inflation_factor(df2.values,i) for i in range(df2.shape[1])]
# print(vifdf)
# # ---다중공선성---
# #    vif_value
# # 0     10.671
# # 1     -7.580
# # 2     -1.622
# # 3     15.918
# # 4     14.261
# #모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

