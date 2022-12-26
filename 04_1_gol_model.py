import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import matplotlib.pyplot as plt
import statsmodels.api
plt.rc('font',family='malgun gothic')
import seaborn as sns
sns.color_palette()
sns.set_palette("pastel")
import statsmodels.formula.api as smf
# import tensorflow.compat.v2 as tf
import tensorflow as tf

df=pd.read_csv("yongsan2021.csv")
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

# print(gol.shape) #(2809, 47) -> (2518, 47)
# print(gol.info()) # 6,18,22,31,20,19
x=gol[['월요일_매출_금액','금요일_매출_금액','남성_매출_금액','수요일_매출_금액','화요일_매출_금액']]
y=gol['분기당_매출_금액']


#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)

import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=gol).fit()
print(lm.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.947
# Model:                            OLS   Adj. R-squared:                  0.947
# Method:                 Least Squares   F-statistic:                 2.875e+04
# Date:                Mon, 26 Dec 2022   Prob (F-statistic):               0.00
# Time:                        15:00:09   Log-Likelihood:            -1.5352e+05
# No. Observations:                8013   AIC:                         3.071e+05
# Df Residuals:                    8007   BIC:                         3.071e+05
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   4.358e+16   7.67e+05   5.68e+10      0.000    4.36e+16    4.36e+16
# x[0]           1.0756      0.010    108.951      0.000       1.056       1.095
# x[1]           1.1586      0.011    109.698      0.000       1.138       1.179
# x[2]           0.0907      0.005     19.716      0.000       0.082       0.100
# x[3]           1.1684      0.011    106.581      0.000       1.147       1.190
# x[4]           1.1356      0.011    106.336      0.000       1.115       1.157
# ==============================================================================
# Omnibus:                     4140.893   Durbin-Watson:                   2.082
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            33445.629
# Skew:                           2.357   Prob(JB):                         0.00
# Kurtosis:                      11.829   Cond. No.                     2.81e+08
# ==============================================================================

print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np
#이작업은 윤현성이 다 하는거야 작업까지 다 끝나고 직원들한테 나눠줘요 그럼 직원들은 predict만 하믄됩니다.
#잔차 먼저 얻어줄게요
df_lm=gol.iloc[:,[6,18,22,31,20,19]]
fitted=lm.predict(df_lm) #이얏 예측값을 얻겠지
residual=df_lm['분기당_매출_금액']-fitted #잔차
print(residual.head(3))
print('잔차의 평균:', np.mean(residual)) 
# 잔차의 평균: -6.983069979642667e-09


print('예측값: ',fitted.values[:3])
print('실제값: ', y.values[:3])

#===============================================================================
print('---선형성---') # 불만족
#===============================================================================
sns.regplot(fitted,residual,lowess=True, line_kws={'color':'yellow'})
plt.plot([fitted.min(),fitted.max()],[0,0],'--')
plt.show()

#===============================================================================
print('---정규성---') # 불만족
#===============================================================================
import scipy.stats as stats
sr=stats.zscore(residual)
(x,y),_=stats.probplot(sr)
sns.scatterplot(x,y)
plt.plot([-3,3],[-3,3],'--')
plt.show()
#찰-싹! 붙어있죠. 잔차항이 정규분포를 따름
#shapiro도 볼 수 있다. 0.05보다 커야해요
print('shapito test: ',stats.shapiro(residual))
# ShapiroResult(statistic=0.8159801959991455, pvalue=0.0)
# pvalue=0.0 얘만 관심있다. < 0.05 정규성불만족

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson:1.834

#===============================================================================
print('---등분산성---') # 불만족
#===============================================================================
#오차들의 분산은 일정해야해
sr=stats.zscore(residual)
sns.regplot(fitted,np.sqrt(abs(sr)),lowess=True, line_kws={'color':'red'})
plt.show()
#평펴엉~합니다 등분산성 만족이에요
#평균선을 기준으로 일정한 패턴을 보이지 않아 등분산성 만족이야

#===============================================================================
print('---다중공선성---')
#===============================================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor
df2=gol[['월요일_매출_금액','금요일_매출_금액','남성_매출_금액','수요일_매출_금액','화요일_매출_금액']]
# print(df2.head(2))
# print(df2.shape) #(2809, 6)
#분산팽창계수를 사용하도록 할게요
vifdf=pd.DataFrame()
vifdf['vif_value']=[variance_inflation_factor(df2.values,i) for i in range(df2.shape[1])]
print(vifdf)

# ---다중공선성---
#    vif_value
# 0      1.744
# 1      2.014
# 2      2.181
# 3      1.926
# 4      1.878

#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

#모델저장
# import pickle
# pickle.dump(lm, open('gol_model.pickle',mode='wb'))



