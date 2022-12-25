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

def predx(tradingArea,smallBusiType):
    #골목/발달/전통/관광 
    sang = bal[bal['상권_코드_명']==tradingArea]
    # 선택한 업종이 있으면 데이터를 불러오고 없으면 0을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):
        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명']==smallBusiType]
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
                              '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1]},index = [str(i)+'분기'])
            
            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata=pd.concat([predictdata,df])
            # 계속돈다.
            i+1
            
        s19=service[service['기준_년_코드']==2019]
        s20=service[service['기준_년_코드']==2020]
        s21=service[service['기준_년_코드']==2021]
        
        return predictdata,s19,s20,s21
    else:
        return 0
data=predx("신용산역(용산역)","중식음식점")
print(data[1])
print(data[1]['점포수'].tolist())
print((data[1][['남성_매출_비율','여성_매출_비율']].values()).tolist())
# # 점포수
# jum_2021_4 = list(bal_4[bal_4['기준_년_코드']=='2021-4']['점포수'])
# # 남녀 매출
# gen_2021_4 = (bal_4[bal_4['기준_년_코드']=='2021-4'][['남성_매출_비율','여성_매출_비율']].values[0]).tolist()
# # 시간대별 매출
# time_2021_4 = (bal_4[bal_4['기준_년_코드']=='2021-4'][['시간대_00_06_매출_비율','시간대_06_11_매출_비율','시간대_11_14_매출_비율','시간대_14_17_매출_비율','시간대_17_21_매출_비율','시간대_21_24_매출_비율']].values[0]).tolist()



# x=bal[['월요일_매출_금액','토요일_매출_금액','일요일_매출_금액','월요일_매출_건수','토요일_매출_건수','일요일_매출_건수']]
# y=bal['분기당_매출_금액']
# model =smf.ols(formula='y ~ x', data=gol).fit()
#
# pred1 = model.predict(pdata.iloc[0])
# pred2 = model.predict(pdata.iloc[1])
# pred3 = model.predict(pdata.iloc[2])
# pred4 = model.predict(pdata.iloc[3])
# print(pred1)
# result1 = (pred1/ xdata['점포수'].iloc[0]).values[0] 
# result2 = (pred2/ xdata['점포수'].iloc[1]).values[0] 
# result3 = (pred3/ xdata['점포수'].iloc[2]).values[0] 
# result4 = (pred4/ xdata['점포수'].iloc[3]).values[0]

#
# # 요청을 받으면 해당 상권에 맞는 코드 실행하기
# # 발달상권 예측모델 실행
# if BigTradingArea == '발달상권':
#     #발달상권
#     print('발달상권 매출 예상')
#     sang = bal[bal['상권_코드_명']==tradingArea]
#
#     if smallBusiType in list(sang['서비스_업종_코드_명']):
#         service = sang[sang['서비스_업종_코드_명']==smallBusiType] 
#
#         xdata=service.groupby(service['기준_분기_코드']).mean()
#
#         data=pd.DataFrame()
#         for i in range(1,5):
#             df = pd.DataFrame({'월요일_매출_금액':xdata['월요일_매출_금액'].iloc[i-1],
#                               '토요일_매출_금액':xdata['토요일_매출_금액'].iloc[i-1],
#                               '일요일_매출_금액':xdata['일요일_매출_금액'].iloc[i-1],
#                               '월요일_매출_건수':xdata['월요일_매출_건수'].iloc[i-1],
#                               '토요일_매출_건수':xdata['토요일_매출_건수'].iloc[i-1],
#                               '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1]},index = [str(i)+'분기'])
#             data=pd.concat([data,df])
#             i+1
#         print(data)
#         print(data.iloc[1])
#
#         # 모델
#         model_bal = smf.ols(formula = '분기당_매출_금액 ~ 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service).fit()
#
#         pred1 = model_bal.predict(data.iloc[0])
#         pred2 = model_bal.predict(data.iloc[1])
#         pred3 = model_bal.predict(data.iloc[2])
#         pred4 = model_bal.predict(data.iloc[3])
#
#         print('발달 실제값:',(xdata['분기당_매출_금액'].mean()) / xdata['점포수'].iloc[0])
#         print('발달 예측값:',pred1 / xdata['점포수'].iloc[0])
#         result1 = (pred1/ xdata['점포수'].iloc[0]).values[0] 
#         result2 = (pred2/ xdata['점포수'].iloc[1]).values[0] 
#         result3 = (pred3/ xdata['점포수'].iloc[2]).values[0] 
#         result4 = (pred4/ xdata['점포수'].iloc[3]).values[0]
#
#         print(model_bal.summary())
#     else : 
#         result1 = 0
#         result2 = 0
#         result3 = 0
#         result4 = 0
#
# # 관광특구 예측모델 실행
# elif BigTradingArea == '관광특구':
#
#     print('관광특구 매출 예상')
#     sang = cul[cul['상권_코드_명']==tradingArea]
#     # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출
#
#     if smallBusiType in list(sang['서비스_업종_코드_명']):
#         service = sang[sang['서비스_업종_코드_명']==smallBusiType] 
#
#         xdata=service.groupby(service['기준_분기_코드']).mean()
#
#         data=pd.DataFrame()
#         for i in range(1,5):
#             df = pd.DataFrame({'월요일_매출_금액':xdata['월요일_매출_금액'].iloc[i-1],
#                               '토요일_매출_금액':xdata['토요일_매출_금액'].iloc[i-1],
#                               '일요일_매출_금액':xdata['일요일_매출_금액'].iloc[i-1],
#                               '월요일_매출_건수':xdata['월요일_매출_건수'].iloc[i-1],
#                               '토요일_매출_건수':xdata['토요일_매출_건수'].iloc[i-1],
#                               '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1]},index = [str(i)+'분기'])
#             data=pd.concat([data,df])
#             i+1
#         print(data)
#         print(data.iloc[1])
#
#         # 모델
#         model = smf.ols(formula = '분기당_매출_금액 ~ 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service).fit()
#
#         pred1 = model.predict(data.iloc[0])
#         pred2 = model.predict(data.iloc[1])
#         pred3 = model.predict(data.iloc[2])
#         pred4 = model.predict(data.iloc[3])
#
#         print('발달 실제값:',(xdata['분기당_매출_금액'].mean()) / xdata['점포수'].iloc[0])
#         print('발달 예측값:',pred1 / xdata['점포수'].iloc[0])
#         result1 = (pred1/ xdata['점포수'].iloc[0]).values[0] 
#         result2 = (pred2/ xdata['점포수'].iloc[1]).values[0] 
#         result3 = (pred3/ xdata['점포수'].iloc[2]).values[0] 
#         result4 = (pred4/ xdata['점포수'].iloc[3]).values[0]
#
#         print(model.summary())
#     else : 
#         result1 = 0
#         result2 = 0
#         result3 = 0
#         result4 = 0
#
# # 골목상권 예측모델 실행 
# elif BigTradingArea == '골목상권':
#
#     print('골목상권 매출 예상')
#     sang = gol[gol['상권_코드_명']==tradingArea]
#     # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출
#
#     if smallBusiType in list(sang['서비스_업종_코드_명']):
#         service = sang[sang['서비스_업종_코드_명']==smallBusiType] 
#
#         xdata=service.groupby(service['기준_분기_코드']).mean()
#
#         data=pd.DataFrame()
#         for i in range(1,5):
#             df = pd.DataFrame({'월요일_매출_금액':xdata['월요일_매출_금액'].iloc[i-1],
#                               '토요일_매출_금액':xdata['토요일_매출_금액'].iloc[i-1],
#                               '일요일_매출_금액':xdata['일요일_매출_금액'].iloc[i-1],
#                               '월요일_매출_건수':xdata['월요일_매출_건수'].iloc[i-1],
#                               '토요일_매출_건수':xdata['토요일_매출_건수'].iloc[i-1],
#                               '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1]},index = [str(i)+'분기'])
#             data=pd.concat([data,df])
#             i+1
#         print(data)
#         print(data.iloc[1])
#
#         # 모델
#         model_bal = smf.ols(formula = '분기당_매출_금액 ~ 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service).fit()
#
#         pred1 = model.predict(data.iloc[0])
#         pred2 = model.predict(data.iloc[1])
#         pred3 = model.predict(data.iloc[2])
#         pred4 = model.predict(data.iloc[3])
#
#         print('발달 실제값:',(xdata['분기당_매출_금액'].mean()) / xdata['점포수'].iloc[0])
#         print('발달 예측값:',pred1 / xdata['점포수'].iloc[0])
#         result1 = (pred1/ xdata['점포수'].iloc[0]).values[0] 
#         result2 = (pred2/ xdata['점포수'].iloc[1]).values[0] 
#         result3 = (pred3/ xdata['점포수'].iloc[2]).values[0] 
#         result4 = (pred4/ xdata['점포수'].iloc[3]).values[0]
#
#         print(model.summary())
#     else : 
#         result1 = 0
#         result2 = 0
#         result3 = 0
#         result4 = 0
#
# # 전통시장 예측모델 실행
# elif BigTradingArea == '전통시장':
#
#     print('전통시장 매출 예상')
#     sang = jeon[jeon['상권_코드_명']==tradingArea]
#     # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출
#
#     if smallBusiType in list(sang['서비스_업종_코드_명']):
#         service = sang[sang['서비스_업종_코드_명']==smallBusiType] 
#
#         xdata=service.groupby(service['기준_분기_코드']).mean()
#
#         data=pd.DataFrame()
#         for i in range(1,5):
#             df = pd.DataFrame({'월요일_매출_금액':xdata['월요일_매출_금액'].iloc[i-1],
#                               '토요일_매출_금액':xdata['토요일_매출_금액'].iloc[i-1],
#                               '일요일_매출_금액':xdata['일요일_매출_금액'].iloc[i-1],
#                               '월요일_매출_건수':xdata['월요일_매출_건수'].iloc[i-1],
#                               '토요일_매출_건수':xdata['토요일_매출_건수'].iloc[i-1],
#                               '일요일_매출_건수':xdata['일요일_매출_건수'].iloc[i-1]},index = [str(i)+'분기'])
#             data=pd.concat([data,df])
#             i+1
#         print(data)
#         print(data.iloc[1])
#
#         # 모델
#         model = smf.ols(formula = '분기당_매출_금액 ~ 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service).fit()
#
#         pred1 = model.predict(data.iloc[0])
#         pred2 = model.predict(data.iloc[1])
#         pred3 = model.predict(data.iloc[2])
#         pred4 = model.predict(data.iloc[3])
#
#         print('발달 실제값:',(xdata['분기당_매출_금액'].mean()) / xdata['점포수'].iloc[0])
#         print('발달 예측값:',pred1 / xdata['점포수'].iloc[0])
#         result1 = (pred1/ xdata['점포수'].iloc[0]).values[0] 
#         result2 = (pred2/ xdata['점포수'].iloc[1]).values[0] 
#         result3 = (pred3/ xdata['점포수'].iloc[2]).values[0] 
#         result4 = (pred4/ xdata['점포수'].iloc[3]).values[0]
#
#         print(model.summary())
#     else : 
#         result1 = 0
#         result2 = 0
#         result3 = 0
#         result4 = 0
#
#
#
# context = {
#     # 'businessType':businessType,
#     'tradingArea':tradingArea,
#     'BigTradingArea':BigTradingArea,
#     'smallBusiType':smallBusiType,
#
#     'result1':result1,
#     'result2':result2,
#     'result3':result3,
#     'result4':result4,
#
#
# }
#
# print(context)
