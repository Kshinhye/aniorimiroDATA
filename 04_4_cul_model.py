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

#temp와 atemp는 상관관계가 높은 것 같다. 긍분산성, 다중분산성 문제가 발생할 가능성이 높다. 둘 중 하나만 사용하는것도 고려
# 모델에 사용할 칼럼들만 가져오도록 한다.
df=pd.read_csv("yongsan2021.csv")
cul= df[df['상권_구분_코드_명']=='관광특구']

#===============================================================================
# 이상치 제거
#===============================================================================
q1=cul['분기당_매출_금액'].quantile(0.25)
q2=cul['분기당_매출_금액'].quantile(0.5)
q3=cul['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
# print(jdf)

condition=cul['분기당_매출_금액']>q3+1.5*iqr
# print(data[condition])

a=cul[condition].index
cul.drop(a,inplace=True)

# print(cul.info()) # 5,21,12,18, 9,19
print(cul.shape) #(181, 47) -> (157, 47)
 
x=cul[['남성_매출_금액','금요일_매출_금액','시간대_14~17_매출_금액','화요일_매출_금액','시간대_17~21_매출_금액']]
y=cul['분기당_매출_금액']

#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)
#합습할 때 fit()의 파라미터값이 중요한것은 딥러닝이다.
import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=cul).fit()
print(lm.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.314
# Model:                            OLS   Adj. R-squared:                  0.291
# Method:                 Least Squares   F-statistic:                     13.80
# Date:                Sat, 24 Dec 2022   Prob (F-statistic):           4.22e-11
# Time:                        01:55:37   Log-Likelihood:                -2900.6
# No. Observations:                 157   AIC:                             5813.
# Df Residuals:                     151   BIC:                             5832.
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   2.361e+07   2.45e+06      9.652      0.000    1.88e+07    2.84e+07
# x[0]           0.0021      0.029      0.074      0.941      -0.054       0.059
# x[1]           0.0263      0.008      3.223      0.002       0.010       0.042
# x[2]           0.0004      0.009      0.040      0.968      -0.018       0.019
# x[3]          -0.0161      0.008     -2.086      0.039      -0.031      -0.001
# x[4]           0.1433      0.031      4.619      0.000       0.082       0.205
# ==============================================================================
# Omnibus:                       54.974   Durbin-Watson:                   1.800
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              117.830
# Skew:                           1.569   Prob(JB):                     2.59e-26
# Kurtosis:                       5.857   Cond. No.                     3.04e+09
# ==============================================================================


print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np
#이작업은 윤현성이 다 하는거야 작업까지 다 끝나고 직원들한테 나눠줘요 그럼 직원들은 predict만 하믄됩니다.
#잔차 먼저 얻어줄게요

df_lm=cul.iloc[:,[5,20,15,12,7,10]]

fitted=lm.predict(df_lm) #이얏 예측값을 얻겠지
residual=df_lm['분기당_매출_금액']-fitted #잔차
print(residual.head(3))
print('잔차의 평균:', np.mean(residual))  #잔차의 평균: 4.176121608466859e-09

#===============================================================================
print('---선형성---') # 불만족
#===============================================================================
sns.regplot(fitted,residual,lowess=True,line_kws={'color':'yellow'})
plt.plot([fitted.min(),fitted.max()],[0,0],'--',color='red')
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
# ShapiroResult(statistic=0.8486631512641907, pvalue=1.9574560722324108e-11)
#pvalue=0.0 얘만 관심있다. < 0.05 정규성불만족

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson: 1.900

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
df2=cul[['연령대_40_매출_금액','남성_매출_금액','시간대_14~17_매출_금액','주중_매출_금액','시간대_06~11_매출_금액']]
# print(df2.head(2))
# print(df2.shape) #(2809, 6)
#분산팽창계수를 사용하도록 할게요
vifdf=pd.DataFrame()
vifdf['vif_value']=[variance_inflation_factor(df2.values,i) for i in range(df2.shape[1])]
print(vifdf)

# ---다중공선성---
#    vif_value
# 0     -7.195
# 1     -0.366
# 2     -0.423
# 3     -0.724
# 4      2.168


#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

