import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
import seaborn as sns
sns.color_palette()
sns.set_palette("pastel")

df=pd.read_csv("yongsan2021.csv")
gol= df[df['상권_구분_코드_명']=='골목상권']
# print(gol.shape) #(8792, 64)

#===============================================================================
# 이상치 제거
#===============================================================================
q1=gol['분기당_매출_금액'].quantile(0.25)
q2=gol['분기당_매출_금액'].quantile(0.5)
q3=gol['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
print(iqr)
condition=gol['분기당_매출_금액']>q3+1.5*iqr
a=gol[condition].index
gol.drop(a,inplace=True)

# print(gol.shape) #(7769, 64))
# print(gol.info())
x=gol[['월요일_매출_금액','금요일_매출_금액','남성_매출_금액','수요일_매출_금액','화요일_매출_금액']]
y=gol['분기당_매출_금액']


import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=gol).fit()
print(lm.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.955
# Model:                            OLS   Adj. R-squared:                  0.955
# Method:                 Least Squares   F-statistic:                 3.315e+04
# Date:                                   Prob (F-statistic):               0.00
# Time:                                   Log-Likelihood:            -1.4485e+05
# No. Observations:                7769   AIC:                         2.897e+05
# Df Residuals:                    7763   BIC:                         2.898e+05
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   4.312e+06   4.48e+05      9.630      0.000    3.43e+06    5.19e+06
# x[0]           0.8624      0.021     41.949      0.000       0.822       0.903
# x[1]           1.6876      0.025     66.500      0.000       1.638       1.737
# x[2]           0.4906      0.010     49.639      0.000       0.471       0.510
# x[3]           1.0679      0.022     47.888      0.000       1.024       1.112
# x[4]           1.1860      0.024     49.474      0.000       1.139       1.233
# ==============================================================================
# Omnibus:                     2064.912   Durbin-Watson:                   1.816
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):           147964.876
# Skew:                           0.298   Prob(JB):                         0.00
# Kurtosis:                      24.371   Cond. No.                     1.37e+08
# ==============================================================================

print('---회귀분석모형의 적절성---')
import numpy as np
# 잔차확인
fitted=lm.predict(x) #이얏 예측값을 얻겠지
residual=y-fitted #잔차
print(residual.head(3))
print('잔차의 평균:', np.mean(residual)) 
# 잔차의 평균: -1.146673987402482e-07

print('예측값: ',fitted.values[:3])
print('실제값: ', y.values[:3])

#===============================================================================
print('---선형성---') 
#===============================================================================
sns.regplot(fitted,residual,lowess=True, line_kws={'color':'yellow'})
plt.plot([fitted.min(),fitted.max()],[0,0],'--')
plt.show()

#===============================================================================
print('---정규성---')
#===============================================================================
import scipy.stats as stats
sr=stats.zscore(residual)
(x,y),_=stats.probplot(sr)
sns.scatterplot(x,y)
plt.plot([-3,3],[-3,3],'--')
plt.show()

#shapiro 0.05보다 커야한다.
print('shapito test: ',stats.shapiro(residual))
# ShapiroResult(statistic=0.7443183064460754, pvalue=0.0)
# pvalue=0.0 < 0.05 정규성불만족(?)

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson:1.834

#===============================================================================
print('---등분산성---')
#===============================================================================
#오차들의 분산은 일정해야해한다
sr=stats.zscore(residual)
sns.regplot(fitted,np.sqrt(abs(sr)),lowess=True, line_kws={'color':'red'})
plt.show()
#평평하다 등분산성 만족이에요
#평균선을 기준으로 일정한 패턴을 보이지 않으면 등분산성을 만족한다.

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
# 0      3.453
# 1      5.629
# 2      0.831
# 3      4.157
# 4      4.550
#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

#모델저장
# import pickle
# pickle.dump(lm, open('gol_model.pickle',mode='wb'))



