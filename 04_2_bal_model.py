import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
import seaborn as sns
sns.color_palette()
sns.set_palette("pastel")

# 모델에 사용할 칼럼들만 가져오도록 한다.
df=pd.read_csv("yongsan2021.csv")
bal= df[df['상권_구분_코드_명']=='발달상권']
# print(bal.shape) #(3593, 64)
#===============================================================================
# 이상치 제거
#===============================================================================
q1=bal['분기당_매출_금액'].quantile(0.25)
q2=bal['분기당_매출_금액'].quantile(0.5)
q3=bal['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
condition=bal['분기당_매출_금액']>q3+1.5*iqr

a=bal[condition].index
bal.drop(a,inplace=True)
# print(bal.shape) #(3593, 64) -> (3123, 64)

x=bal[['시간대_14_17_매출_금액','수요일_매출_금액','시간대_11_14_매출_금액','월요일_매출_금액','금요일_매출_금액']]
y=bal['분기당_매출_금액']

import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=bal).fit()
print(lm.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.938
# Model:                            OLS   Adj. R-squared:                  0.938
# Method:                 Least Squares   F-statistic:                     9431.
# Date:                                   Prob (F-statistic):               0.00
# Time:                                   Log-Likelihood:                -62766.
# No. Observations:                3123   AIC:                         1.255e+05
# Df Residuals:                    3117   BIC:                         1.256e+05
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     2.2e+07   2.95e+06      7.461      0.000    1.62e+07    2.78e+07
# x[0]           0.3861      0.022     17.420      0.000       0.343       0.430
# x[1]           0.9778      0.048     20.195      0.000       0.883       1.073
# x[2]           0.3148      0.020     15.692      0.000       0.275       0.354
# x[3]           1.1624      0.049     23.779      0.000       1.067       1.258
# x[4]           2.6465      0.050     53.002      0.000       2.549       2.744
# ==============================================================================
# Omnibus:                      937.338   Durbin-Watson:                   1.789
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            32332.568
# Skew:                           0.756   Prob(JB):                         0.00
# Kurtosis:                      18.690   Cond. No.                     4.02e+08
# ==============================================================================

print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np
print(bal.info())

pred=lm.predict(x) 
print('예측값: ', pred[:3])
print('실제값: ', y[:3])
residual=y-pred #잔차
# print(residual.head(3))
print('잔차의 평균:', np.mean(residual))  #잔차의 평균: -2.730018696340658e-07

#===============================================================================
print('---선형성---')
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

#shapiro test
print('shapito test: ',stats.shapiro(residual))
# ShapiroResult(statistic=0.7529749870300293, pvalue=0.0)
# pvalue=0.0 < 0.05 정규성불만족

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson:  1.834

#===============================================================================
print('---등분산성---') 
#===============================================================================
#오차들의 분산은 일정해야해
sr=stats.zscore(residual)
sns.regplot(pred,np.sqrt(abs(sr)),lowess=True,  line_kws={'color':'yellow'})
plt.show()

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
#  0    -0.082
# 1      0.042
# 2      0.179
# 3     -0.118
# 4      1.125

#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

# #모델저장
# import pickle
# pickle.dump(lm, open('bal_model.pickle',mode='wb'))