import pandas as pd
from sklearn.metrics._regression import r2_score
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

df=pd.read_csv("yongsan2021.csv")
jdf= df[df['상권_구분_코드_명']=='전통시장']
print(jdf.shape) #(1569, 64)
#===============================================================================
# 이상치 제거
#===============================================================================
q1=jdf['분기당_매출_금액'].quantile(0.25)
q2=jdf['분기당_매출_금액'].quantile(0.5)
q3=jdf['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
condition=jdf['분기당_매출_금액']>q3+1.5*iqr
a=jdf[condition].index
jdf.drop(a,inplace=True)
print(jdf.shape) #(1569, 64) -> (1413, 64)
 
x=jdf[['금요일_매출_금액','수요일_매출_금액','여성_매출_금액','화요일_매출_금액','목요일_매출_금액']]
y=jdf['분기당_매출_금액']

import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=jdf).fit()
print(lm.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.964
# Model:                            OLS   Adj. R-squared:                  0.964
# Method:                 Least Squares   F-statistic:                     7792.
# Date:                Mon, 26 Dec 2022   Prob (F-statistic):               0.00
# Time:                        15:04:06   Log-Likelihood:                -27968.
# No. Observations:                1453   AIC:                         5.595e+04
# Df Residuals:                    1447   BIC:                         5.598e+04
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   4.358e+16   1.98e+06    2.2e+10      0.000    4.36e+16    4.36e+16
# x[0]           1.0072      0.026     38.800      0.000       0.956       1.058
# x[1]           1.2371      0.026     48.034      0.000       1.187       1.288
# x[2]           0.1585      0.011     14.022      0.000       0.136       0.181
# x[3]           1.1138      0.024     45.609      0.000       1.066       1.162
# x[4]           1.1170      0.026     42.507      0.000       1.065       1.169
# ==============================================================================
# Omnibus:                      788.258   Durbin-Watson:                   1.965
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             8801.179
# Skew:                           2.295   Prob(JB):                         0.00
# Kurtosis:                      14.149   Cond. No.                     3.34e+08
# ==============================================================================

print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np

pred=lm.predict(x)
print('예측값: ',pred.values[:3])
print('실제값: ', y.values[:3])
residual=y-pred #잔차


print(residual.head(3))
print('잔차의 평균:', np.mean(residual))  #잔차의 평균: 잔차의 평균: -4.1238816242257395

#===============================================================================
print('---선형성---') # 불만족
#===============================================================================
sns.regplot(pred,residual,lowess=True, line_kws={'color':'yellow'})
plt.plot([pred.min(),pred.max()],[0,0],'--')
plt.show()
#잔차가 일정하게 분포되어있으므로 선형성 만족
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
# shapito test:  ShapiroResult(statistic=0.7646639943122864, pvalue=6.353347107402288e-41)
#pvalue=0.0 얘만 관심있다. < 0.05 정규성불만족

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson: 1.909

#===============================================================================
print('---등분산성---') # 불만족
#===============================================================================
#오차들의 분산은 일정해야해
sr=stats.zscore(residual)
sns.regplot(pred,np.sqrt(abs(sr)),lowess=True, line_kws={'color':'red'})
plt.show()
#평펴엉~합니다 등분산성 만족이에요
#평균선을 기준으로 일정한 패턴을 보이지 않아 등분산성 만족이야

#===============================================================================
print('---다중공선성---')
#===============================================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor

#분산팽창계수를 사용
vifdf=pd.DataFrame()
vifdf['vif_value']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
print(vifdf)
# ---다중공선성---
#    vif_value
# 0      5.735
# 1      2.685
# 2      0.145
# 3      5.531
# 4      5.811

#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

#모델저장
# import pickle
# pickle.dump(lm, open('j_model.pickle',mode='wb'))