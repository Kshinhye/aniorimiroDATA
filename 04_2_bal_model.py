import pandas as pd
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
bal= df[df['상권_구분_코드_명']=='발달상권']

# print(bal.info()) # 5  31  24  39  36   30
# print(bal.shape) #(1165, 47) -> (1044, 47)

x=bal[['일요일_매출_건수','주말_매출_건수','여성_매출_건수','시간대_건수~21_매출_건수','토요일_매출_건수']]
y=bal['분기당_매출_금액']

#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)
#합습할 때 fit()의 파라미터값이 중요한것은 딥러닝이다.
import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=bal).fit()
print(lm.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.277
# Model:                            OLS   Adj. R-squared:                  0.274
# Method:                 Least Squares   F-statistic:                     111.0
# Date:                Sat, 24 Dec 2022   Prob (F-statistic):           3.94e-80
# Time:                        16:10:56   Log-Likelihood:                -24239.
# No. Observations:                1165   AIC:                         4.849e+04
# Df Residuals:                    1160   BIC:                         4.851e+04
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    3.89e+07   8.19e+06      4.749      0.000    2.28e+07     5.5e+07
# x[0]        1.222e+04   1080.939     11.306      0.000    1.01e+04    1.43e+04
# x[1]        2610.2479    506.111      5.157      0.000    1617.252    3603.244
# x[2]         451.0191    417.896      1.079      0.281    -368.897    1270.935
# x[3]         711.0114   1014.353      0.701      0.483   -1279.160    2701.182
# x[4]       -9610.8612   1136.783     -8.454      0.000   -1.18e+04   -7380.481
# ==============================================================================
# Omnibus:                     1510.302   Durbin-Watson:                   1.834
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):           331894.283
# Skew:                           6.750   Prob(JB):                         0.00
# Kurtosis:                      84.578   Cond. No.                     3.84e+15
# ==============================================================================

print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np

df_lm=bal.iloc[:,[5,31,24,39,36,30]]

pred=lm.predict(df_lm) #이얏 예측값을 얻겠지
print('예측값: ', pred[:3])
print('실제값: ', y[:3])
residual=df_lm['분기당_매출_금액']-pred #잔차
# print(residual.head(3))
print('잔차의 평균:', np.mean(residual))  #잔차의 평균: 1.0232557042985515e-07

#===============================================================================
print('---선형성---') # 불만족
#===============================================================================
sns.regplot(pred,residual,lowess=True, line_kws={'color':'red'})
plt.plot([pred.min(),pred.max()],[0,0],'--',color='blue')
plt.show()
#잔차가 일정하게 분포되어있으므로 선형성 만족
#===============================================================================
print('---정규성---') # 불만족
#===============================================================================
import scipy.stats as stats
sr=stats.zscore(residual)
(x,y),_=stats.probplot(sr)
sns.scatterplot(x,y)
plt.plot([-3,3],[-3,3],'--',color='yellow')
plt.show()
#찰-싹! 붙어있죠. 잔차항이 정규분포를 따름
#shapiro도 볼 수 있다. 0.05보다 커야해요
print('shapito test: ',stats.shapiro(residual))
# ShapiroResult(statistic=0.8828872442245483, pvalue=2.3844260714117943e-27)
# pvalue=0.0 얘만 관심있다. < 0.05 정규성불만족

#===============================================================================
print('---독립성---')
#===============================================================================
#Durbin-Watson:  1.834

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
df2=bal[['일요일_매출_건수','주말_매출_건수','여성_매출_건수','시간대_건수~21_매출_건수','토요일_매출_건수']]
# print(df2.head(2))
# print(df2.shape) #(2809, 6)
#분산팽창계수를 사용하도록 할게요
vifdf=pd.DataFrame()
vifdf['vif_value']=[variance_inflation_factor(df2.values,i) for i in range(df2.shape[1])]
print(vifdf)
# ---다중공선성---
#    vif_value
# 0      0.176
# 1     -0.475
# 2     -0.271
# 3     -0.149
# 4     -0.145
# 5     -0.492

#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

