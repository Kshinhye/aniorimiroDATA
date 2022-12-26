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
bal= df[df['상권_구분_코드_명']=='발달상권']

#===============================================================================
# 이상치 제거
#===============================================================================
q1=bal['분기당_매출_금액'].quantile(0.25)
q2=bal['분기당_매출_금액'].quantile(0.5)
q3=bal['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
# print(jdf)

condition=bal['분기당_매출_금액']>q3+1.5*iqr
# print(data[condition])

a=bal[condition].index
bal.drop(a,inplace=True)

# print(bal.info()) # 6,11,20,27,18,22
# print(bal.shape) #(1165, 47) -> (1044, 47)

x=bal[['시간대_14_17_매출_금액','수요일_매출_금액','시간대_11_14_매출_금액','월요일_매출_금액','금요일_매출_금액']]
y=bal['분기당_매출_금액']

#적절성이 만족도에 영향을 준다라는 가정하에 모델 생성(사람이 생각한거 정말로 확인하려면 p값 확인해야함)
#합습할 때 fit()의 파라미터값이 중요한것은 딥러닝이다.
import statsmodels.formula.api as smf
lm=smf.ols(formula='y ~ x', data=bal).fit()
print(lm.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.954
# Model:                            OLS   Adj. R-squared:                  0.954
# Method:                 Least Squares   F-statistic:                 1.317e+04
# Date:                Mon, 26 Dec 2022   Prob (F-statistic):               0.00
# Time:                        15:01:57   Log-Likelihood:                -63104.
# No. Observations:                3181   AIC:                         1.262e+05
# Df Residuals:                    3175   BIC:                         1.263e+05
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   4.358e+16   2.43e+06    1.8e+10      0.000    4.36e+16    4.36e+16
# x[0]           0.0547      0.011      4.763      0.000       0.032       0.077
# x[1]           1.3464      0.027     49.152      0.000       1.293       1.400
# x[2]           0.1821      0.011     15.845      0.000       0.160       0.205
# x[3]           1.4283      0.027     52.559      0.000       1.375       1.482
# x[4]           1.5576      0.026     59.870      0.000       1.507       1.609
# ==============================================================================
# Omnibus:                     1074.174   Durbin-Watson:                   1.954
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            40234.701
# Skew:                           0.916   Prob(JB):                         0.00
# Kurtosis:                      20.326   Cond. No.                     5.36e+08
# ==============================================================================

print('---회귀분석모형의 적절성 확인 작업을 해봅시다---')
import numpy as np
print(bal.info())
df_lm=bal.iloc[:,[6,11,20,27,18,22]]

pred=lm.predict(df_lm) #이얏 예측값을 얻겠지
print('예측값: ', pred[:3])
print('실제값: ', y[:3])
residual=df_lm['분기당_매출_금액']-pred #잔차
# print(residual.head(3))
print('잔차의 평균:', np.mean(residual))  #잔차의 평균: -7.14995284501729

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
# ShapiroResult(statistic=0.776547908782959, pvalue=0.0)
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
sns.regplot(pred,np.sqrt(abs(sr)),lowess=True,  line_kws={'color':'yellow'})
plt.show()
#평펴엉~합니다 등분산성 만족이에요
#평균선을 기준으로 일정한 패턴을 보이지 않아 등분산성 만족이야

#===============================================================================
print('---다중공선성---')
#===============================================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor
df2=bal[['여성_매출_금액','남성_매출_금액','시간대_11_14_매출_금액','월요일_매출_금액','금요일_매출_금액']]
# print(df2.head(2))
# print(df2.shape) #(2809, 6)
#분산팽창계수를 사용하도록 할게요
vifdf=pd.DataFrame()
vifdf['vif_value']=[variance_inflation_factor(df2.values,i) for i in range(df2.shape[1])]
print(vifdf)
# ---다중공선성---
#    vif_value
# 0      3.671
# 1      4.214
# 2      2.819
# 3      3.867
# 4      5.792

#모든 변수가 10을 넘기지 않음, 다중공선성이 발생하지 않음(다중공선성 우려 없음)

#모델저장
import pickle
pickle.dump(lm, open('bal_model.pickle',mode='wb'))