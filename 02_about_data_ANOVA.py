#===============================================================================
# ANOVA TEST - 상권구분
#===============================================================================
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
df=pd.read_csv("yongsan2021.csv", usecols=['상권_구분_코드_명','분기당_매출_금액','점포수'])
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.
df['분기당_매출_금액']=df['분기당_매출_금액']/df['점포수']
print(df['상권_구분_코드_명'].unique()) #['골목상권' '발달상권' '전통시장' '관광특구']
# 상권은 '골목상권' '발달상권' '전통시장' '관광특구'로 나누어져 있다

#===============================================================================
# 평균 막대 그래프로 그려보기
#===============================================================================
# 네 구역의 상권별로 매출금액에 차이가 있을까? 구역별로 차이가 있다면 모델링에 차별화가 필요할 것이다.
print('골목상권 평균',df[df['상권_구분_코드_명']=='골목상권']['분기당_매출_금액'].mean()) #  1.1201244392124966e+16
print('발달상권 평균',df[df['상권_구분_코드_명']=='발달상권']['분기당_매출_금액'].mean()) #  7764045575357264.0
print('전통시장 평균',df[df['상권_구분_코드_명']=='전통시장']['분기당_매출_금액'].mean()) #  1.1043752667368046e+16
print('관광특구 평균',df[df['상권_구분_코드_명']=='관광특구']['분기당_매출_금액'].mean()) #  8790481961587027

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')  #한글깨짐 방지
sns.color_palette()
sns.set_palette("RdBu", 10)

mean=df.groupby(['상권_구분_코드_명']).mean()['분기당_매출_금액']
sns.barplot(y=mean.index,x=mean)
plt.xticks(size=10)
plt.show()

#===============================================================================
# 선형회귀를 위한 확인
#===============================================================================

# 범주형을 연속형으로 바꾼다.
df['상권_구분_코드_명']=df['상권_구분_코드_명'].map({'골목상권':3,'발달상권':2,'관광특구':1, '전통시장':0 })
# print(df['상권_구분_코드_명'].unique()) #[3 2 0 1]

# 이상치 제거
q1=df['분기당_매출_금액'].quantile(0.25)
q2=df['분기당_매출_금액'].quantile(0.5)
q3=df['분기당_매출_금액'].quantile(0.75)
iqr=q3-q1
# print(iqr)
condition=df['분기당_매출_금액']>q3+1.5*iqr
# print(data[condition])

a=df[condition].index #480 개
df.drop(a,inplace=True)
print(df.shape) #(3750, 2)

GM=df[df['상권_구분_코드_명']==3]['분기당_매출_금액'] #골목상권
BD=df[df['상권_구분_코드_명']==2]['분기당_매출_금액'] #발달상권
GG=df[df['상권_구분_코드_명']==1]['분기당_매출_금액'] #관광특구
JT=df[df['상권_구분_코드_명']==0]['분기당_매출_금액'] #전통시장

# 금액의 단위가 크기때문에 로깅
GM=np.log(GM) 
BD=np.log(BD) 
GG=np.log(GG) 
JT=np.log(JT) 

import scipy.stats as stats
# 정규성확인
print(stats.shapiro(GM).pvalue) # 0.0
print(stats.shapiro(BD).pvalue) # 2.687035992140221e-37
print(stats.shapiro(GG).pvalue) # 5.306251271708362e-11
print(stats.shapiro(JT).pvalue) # 3.635869046624615e-33
print()

# 등분산성 확인
print('levene',stats.levene(GM,BD,GG,JT).pvalue) # 7.045880727319484e-128

# 크루스칼 왈리스
print(stats.kruskal(GM,BD,GG,JT))
# KruskalResult(statistic=855.7850010545635, pvalue=3.445697630853123e-185)

#pip install pingouin
from pingouin import welch_anova
print('welch_anova',welch_anova(data=df, dv='분기당_매출_금액', between='상권_구분_코드_명'))
# welch_anova        Source       ddof1    ddof2       F        p-unc   np2
#     0          상권_구분_코드_명       3    1952.126    289.655   0.000   0.059

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
#ols(최소자승법): 선형회귀 모델을 만듬
#파이썬에서 그룹이 범주형일 때 C() 둘러줘야한다.
lmodel=ols('분기당_매출_금액 ~ C(상권_구분_코드_명)', data=df).fit() #학습해라= 최적의 모델을 만들어라.
print('anova_lm',anova_lm(lmodel,type=1))
# 해석: p-value  6.844746e-13 < 0.05 | 유의함으로 네곳의 상권은 평균차이가 있다.
# 고로 각 상권에 맞도록 매출을 예측하도록 한다.
#                     df        sum_sq       mean_sq          F        PR(>F)
# C(상권_구분_코드_명)      3.0  5.046663e+18  1.682221e+18  56.718359  1.922155e-36


# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=df.분기당_매출_금액, groups=df.상권_구분_코드_명, alpha=0.05) #알파값은 0.05 기본
print(turkeyResult)

turkeyResult.plot_simultaneous(xlabel='mean' , ylabel='group')
plt.show()

#                   Multiple Comparison of Means - Tukey HSD, FWER=0.05                  
# =======================================================================================
# group1 group2       meandiff      p-adj         lower               upper        reject
# ---------------------------------------------------------------------------------------
#      0      1 -3678035766611086.0   -0.0 -4436542900335454.5 -2919528632886717.5   True
#      0      2 -2917180537855892.0   -0.0 -3373295454359584.0 -2461065621352200.0   True
#      0      3   190369891416402.0 0.6385  -223558632207758.3   604298415040562.2  False
#      1      2   760855228755194.0 0.0274  59029919962619.125  1462680537547769.0   True
#      1      3  3868405658027488.0   -0.0  3193235479721085.0  4543575836333891.0   True
#      2      3  3107550429272294.0   -0.0  2809955881572386.0  3405144976972202.0   True
# ---------------------------------------------------------------------------------------
