import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings(action='ignore')# 경고출력안하기
import scipy.stats as stats
# from sklearn.preprocessing import LabelEncoder   
import statsmodels.formula.api as smf

bal=pd.read_csv("https://raw.githubusercontent.com/Kshinhye/aniorimiroDATA/master/yongsan2021.csv", encoding='utf-8')


# BigTradingArea="전통시장"
# tradingArea="용산용문시장"
# smallBusiType="한식음식점"

def balpredx(tradingArea,smallBusiType):
    #골목/발달/전통/관광 
    sang = bal[bal['상권_코드_명']==tradingArea]
    # 선택한 업종이 있으면 데이터를 불러오고 없으면 0을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):
        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명']==smallBusiType]
        
        ###### 모델에 넣어줄 미지의 값이다.(예측값에 사용)
        # 분기별 평균을 구한다.
        xdata=service.groupby(service['기준_분기_코드']).mean()
        # 예측값에 넣을 변수들만 담기위해 빈 데이터 프레임을 만들어준다.
        predictdata=pd.DataFrame()
        # 산정이 안된 분기가 있을 경우 길이를 모르기때문에 index를 돈다.
        for i in xdata.index:
            print(i)
            # 0부터 시작하기때문에 -1을 해준다.
            df = pd.DataFrame({'월요일_매출_금액':xdata['월요일_매출_금액'].iloc[i-1],
                              '토요일_매출_금액':xdata['토요일_매출_금액'].iloc[i-1],
                              '일요일_매출_금액':xdata['일요일_매출_금액'].iloc[i-1],
                              '월요일_매출_건수':xdata['월요일_매출_건수'].iloc[i-1],
                              '토요일_매출_건수':xdata['토요일_매출_건수'].iloc[i-1],
                              '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1],
                              '점포수':xdata['점포수'].iloc[i-1]},
                              index = [str(i)+'분기'])
            
            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata=pd.concat([predictdata,df])
            # 계속돈다.
            i+1
            
        return predictdata
    
    # 해당 상권이 없으면 0을 리턴한다.
    else:
        return 0
data=balpredx("신용산역(용산역)","중식음식점")
print(data)
# import pickle
# model=pickle.load(open('/aniorimiro/static/data/gol_model.sav', mode='rb'))


# print(context)