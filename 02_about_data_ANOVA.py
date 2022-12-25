#===============================================================================
# ANOVA TEST - 상권구분
#===============================================================================
import pandas as pd
import numpy as np

df=pd.read_csv("yongsan2021.csv", usecols=['상권_구분_코드_명','분기당_매출_금액','점포수'])
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.
df['분기당_매출_금액']=df['분기당_매출_금액']/df['점포수']

print(df['상권_구분_코드_명'].unique()) #['골목상권' '발달상권' '전통시장' '관광특구']
# 상권은 '골목상권' '발달상권' '전통시장' '관광특구'로 나누어져 있다.
# 네 구역의 상권별로 매출금액에 차이가 있을까? 구역별로 차이가 있다면 모델링에 차별화가 필요할 것이다.
print(df[df['상권_구분_코드_명']=='골목상권']['분기당_매출_금액'].mean()) #  40078399
print(df[df['상권_구분_코드_명']=='발달상권']['분기당_매출_금액'].mean()) #  82345784
print(df[df['상권_구분_코드_명']=='전통시장']['분기당_매출_금액'].mean()) #  43656244
print(df[df['상권_구분_코드_명']=='관광특구']['분기당_매출_금액'].mean()) # 139787023

mean=df.groupby(['상권_구분_코드_명']).mean()['분기당_매출_금액']

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')  #한글깨짐 방지
sns.color_palette()
sns.set_palette("RdBu", 10)

sns.barplot(y=mean.index,x=mean)
plt.xticks(size=10)
plt.show()
# 범주형을 연속형으로 바꾼다.
df['상권_구분_코드_명']=df['상권_구분_코드_명'].map({'골목상권':3,'발달상권':2,'관광특구':1, '전통시장':0 })
# print(df['상권_구분_코드_명'].unique()) #[3 2 0 1]

#===============================================================================
# 이상치 제거
#===============================================================================
# q1=df['분기당_매출_금액'].quantile(0.25)
# q2=df['분기당_매출_금액'].quantile(0.5)
# q3=df['분기당_매출_금액'].quantile(0.75)
# iqr=q3-q1
# # print(iqr)
#
# condition=df['분기당_매출_금액']>q3+1.5*iqr
# # print(data[condition])
#
# a=df[condition].index #480 개
# df.drop(a,inplace=True)
# print(df.shape) #(3750, 2)

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
print(stats.shapiro(GM).pvalue) # 1.6404176628181943e-10
print(stats.shapiro(BD).pvalue) # 5.8469474921440945e-12
print(stats.shapiro(GG).pvalue) # 0.14129702746868134
print(stats.shapiro(JT).pvalue) # 0.1431194245815277
print()
# 등분산성 확인
print(stats.levene(GM,BD,GG,JT).pvalue)  # 0.0018485609049856873

# 크루스칼 왈리스
print(stats.kruskal(GM,BD,GG,JT))
# KruskalResult(statistic=120.9579320542866, pvalue=4.7986931524027327e-26)

#pip install pingouin
from pingouin import welch_anova
print(welch_anova(data=df, dv='분기당_매출_금액', between='상권_구분_코드_명'))
#        Source  ddof1        ddof2          F         p-unc       np2
# 0  상권_구분_코드_명      3  2019.785223  29.785171  7.703434e-19  0.011599
#                     df        sum_sq       mean_sq          F        PR(>F)
# C(상권_구분_코드_명)      3.0  5.046663e+18  1.682221e+18  56.718359  1.922155e-36


from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
#ols(최소자승법): 선형회귀 모델을 만듬
#파이썬에서 그룹이 범주형일 때 C() 둘러줘야한다.
lmodel=ols('분기당_매출_금액 ~ C(상권_구분_코드_명)', data=df).fit() #학습해라= 최적의 모델을 만들어라.
print(anova_lm(lmodel,type=1))
# 해석: p-value  6.844746e-13 < 0.05 | 유의함으로 네곳의 상권은 평균차이가 있다.
# 고로 각 상권에 맞도록 매출을 예측하도록 한다.


# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=df.분기당_매출_금액, groups=df.상권_구분_코드_명, alpha=0.05) #알파값은 0.05 기본
print(turkeyResult)

turkeyResult.plot_simultaneous(xlabel='mean' , ylabel='group')
plt.show()

#            Multiple Comparison of Means - Tukey HSD, FWER=0.05           
# ========================================================================  
# group1 group2    meandiff    p-adj      lower          upper      reject  
# ------------------------------------------------------------------------  
#      0      1  96130779.1833    0.0   47781878.1161 144479680.2505   True  
#      0      2  38689540.1247 0.0041    9240693.2677  68138386.9816   True  
#      0      3   -3577844.979 0.9858  -30203788.7747  23048098.8167  False  # 전통 골목
#      1      2 -57441239.0587 0.0055 -102288922.8457 -12593555.2716   True
#      1      3 -99708624.1623    0.0  -142755355.082 -56661893.2427   True 
#      2      3 -42267385.1037    0.0  -61828555.5613 -22706214.6461   True 
# ------------------------------------------------------------------------  