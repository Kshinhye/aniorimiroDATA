import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.3f}'.format # 지수표현식이 보기 불편할 때
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')  #한글깨짐 방지

df=pd.read_csv("yongsan2021.csv")
# 현재 파일에는 각 상권의 모든 점포수의 매출금액합계가 들어가있다. 점포당 매출금액을 비교하기 위해서 매출금액을 점포수만큼 나눠준다.

#===============================================================================
# 골목상권 상관관계 확인
#===============================================================================
gol= df[df['상권_구분_코드_명']=='골목상권']

print("골목상권 분기당매출금액 상관계수 \n", gol.corr()['분기당_매출_금액'].sort_values(ascending=False))
gol_corr=gol[['분기당_매출_금액','월요일_매출_금액','금요일_매출_금액','남성_매출_금액','수요일_매출_금액','화요일_매출_금액']]
plt.figure(figsize=(10,10))
sns.heatmap(data = gol_corr.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

#===============================================================================
# 발달상권 상관관계 확인
#===============================================================================
bal= df[df['상권_구분_코드_명']=='발달상권']
# print(bal)

print("발달상권 분기당매출금액 상관계수 \n", bal.corr()['분기당_매출_금액'].sort_values(ascending=False))
# 전체적으로 상관관계가 낮다 
bal_corr=bal[['분기당_매출_금액','시간대_14~17_매출_금액','수요일_매출_금액','시간대_11~14_매출_금액','월요일_매출_금액','금요일_매출_금액']]
plt.figure(figsize=(10,10))
sns.heatmap(data = bal_corr.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

#===============================================================================
# 전통시장 상관관계 확인
#===============================================================================
jdf= df[df['상권_구분_코드_명']=='전통시장']

print("전통시장 분기당매출금액 상관계수 \n", jdf.corr()['분기당_매출_금액'].sort_values(ascending=False))
jdf_corr=jdf[['분기당_매출_금액','금요일_매출_금액','수요일_매출_금액','여성_매출_금액','화요일_매출_금액','목요일_매출_금액']]
plt.figure(figsize=(10,10))
sns.heatmap(data = jdf_corr.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

#===============================================================================
# 관광특구 상관관계 확인
#===============================================================================
cul= df[df['상권_구분_코드_명']=='관광특구']
# print(cul)
print("관광특구 분기당매출금액 상관계수 \n", cul.corr()['분기당_매출_금액'].sort_values(ascending=False))
cul_corr=cul[['분기당_매출_금액','남성_매출_금액','금요일_매출_금액','시간대_14~17_매출_금액','화요일_매출_금액','시간대_17~21_매출_금액']]
plt.figure(figsize=(10,10))
sns.heatmap(data = cul_corr.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()