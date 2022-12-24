# Clustering(군집화) : 사전정보(label)가 없는 자료에 대해 컴퓨터가 스스로 패턴을 찾아 여러개의 군집을 형성함
# 비지도학습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
df=pd.read_csv("yongsan2021.csv")
columns=['기준_분기_코드','상권_구분_코드_명','상권_코드_명','서비스_업종_코드_명',
         '분기당_매출_금액','주중_매출_금액','주말_매출_금액',
         '분기당_매출_건수','주중_매출_건수','주말_매출_건수','월요일_매출_건수','화요일_매출_건수','수요일_매출_건수','목요일_매출_건수','금요일_매출_건수','토요일_매출_건수','일요일_매출_건수',
         '시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액','시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액',
         '시간대_건수~06_매출_건수','시간대_건수~11_매출_건수','시간대_건수~14_매출_건수','시간대_건수~17_매출_건수','시간대_건수~21_매출_건수','시간대_건수~24_매출_건수',
         '연령대_10_매출_금액','연령대_20_매출_금액','연령대_30_매출_금액','연령대_40_매출_금액','연령대_50_매출_금액','연령대_60_이상_매출_금액',
         '연령대_10_매출_건수','연령대_20_매출_건수','연령대_30_매출_건수','연령대_40_매출_건수','연령대_50_매출_건수','연령대_60_이상_매출_건수',
         '남성_매출_금액','여성_매출_금액','남성_매출_건수','여성_매출_건수','점포수'
         ]

# 데이터간 거리보기
# 유클리디안 거리 함수
from scipy.spatial.distance import pdist, squareform
dist_vec=pdist(df, metric='euclidean') #데이터(배열)에 대해 각 요소간 거리를 계산한 후 1차원 배열로 반환해준다
print(dist_vec)

# 보기편하게 squareform에 넣어볼게요
row_dist=pd.DataFrame(squareform(dist_vec), columns=columns, index=df.index)
print(row_dist)

# 우리의 찐 목적은 거리를 보는게 아니라 군집화 하려는거야
# 계층적 군집분석(비지도학습)
from scipy.cluster.hierarchy import linkage
row_clusters=linkage(dist_vec, method='complete') #method(연결방법) complete(완전연결법) (방법의 차이일 뿐 결과는 거의 비슷하다)

df=pd.DataFrame(row_clusters, columns=columns)
print(df)

# dendrogram으로 row_clusters를 시각화
from scipy.cluster.hierarchy import dendrogram
low_dend=dendrogram(row_clusters,labels=df.index)
plt.ylabel('유클리드 거리')
plt.show()