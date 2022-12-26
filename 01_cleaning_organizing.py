import pandas as pd
import numpy as np
# pd.set_option('display.max_columns', None) 
# pd.set_option('display.max_rows', None)

### 상권_구분_코드 ###
# A 골목상권
# U 관광특구
# R 전통시장
# D 발달상권

### 상권_코드 ###
# 1001491
# 2110058 ~ 2110099
# 2120040 ~ 2120047  #2120043은 중구라서 제외
# 2130056 ~ 2130062

### 상권별 ###
# CS100001 ~ CS100010 #외식업 WS
# CS200001 ~ CS200037 #서비스업 SBS
# CS300001 ~ CS300043 #소매업 SM

### 용산구만 사용하기위해 용산 상권들만 가져온다 ###
# 우리마을가게 상권분석서비스 서비스 상권영역 pdf 파일 참고
# 위도 경도로 가져오려 했으나, 검색이 불가한 장소가 있어 정확성을 위해 직접 비교함
# * 상권코드명을 변경할 방법도 있으나 우선 pdf자료를 활용

yongsan =['NH농협은행 보광동지점','경리단길남측','경리단길북측','남영역 1번','리움미술관','배문고등학교','삼각지역 14번','삼각지역 3번','삼광초등학교',
          '새남터성지','서빙고동주민센터','서빙고역 1번','서울독일학교','서울역 12번','서울역 15번','성심여자고등학교','숙대입구','숙대입구역 1번','열정도','오산고등학교','오산중학교',
          '용산구청','용산세무서','우사단길','유엔빌리지길','이촌동점보아파트','이태원엔틱가구거리','이태원역 북측','한강로동땡땡거리(은행나무길)','한강미주맨션아파트','한강진역 3번',
          '한국폴리텍대학서울정수캠퍼스','한남초등학교','한남힐사이드아파트','해방촌 남동측','해방촌예술마을','효창공원앞역 2번','효창공원앞역 5번','효창공원앞역 6번','효창동주민센터','후암동주민센터',
          '이태원 관광특구','삼각지역','남영동 먹자골목','이태원(이태원역)','한남오거리','용산용문시장','만리시장','이촌종합시장','후암시장','신흥시장','이태원시장','보광시장','서울역','숙대입구역(남영역, 남영동)'
          ,'용산전자상가(용산역)','남정초등학교','신용산역(용산역)']

columns=['기준_년_코드','기준_분기_코드','상권_구분_코드_명','상권_코드_명','서비스_업종_코드_명',
         '분기당_매출_금액','주중_매출_금액','주말_매출_금액','월요일_매출_금액','화요일_매출_금액','수요일_매출_금액','목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액',
         '분기당_매출_건수','주중_매출_건수','주말_매출_건수','월요일_매출_건수','화요일_매출_건수','수요일_매출_건수','목요일_매출_건수','금요일_매출_건수','토요일_매출_건수','일요일_매출_건수',
         '시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액','시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액',
         '시간대_건수~06_매출_건수','시간대_건수~11_매출_건수','시간대_건수~14_매출_건수','시간대_건수~17_매출_건수','시간대_건수~21_매출_건수','시간대_건수~24_매출_건수',
         '연령대_10_매출_금액','연령대_20_매출_금액','연령대_30_매출_금액','연령대_40_매출_금액','연령대_50_매출_금액','연령대_60_이상_매출_금액',
         '연령대_10_매출_건수','연령대_20_매출_건수','연령대_30_매출_건수','연령대_40_매출_건수','연령대_50_매출_건수','연령대_60_이상_매출_건수',
         '남성_매출_금액','여성_매출_금액','남성_매출_건수','여성_매출_건수','점포수',
         '남성_매출_비율','여성_매출_비율','시간대_00~06_매출_비율','시간대_06~11_매출_비율','시간대_11~14_매출_비율','시간대_14~17_매출_비율','시간대_17~21_매출_비율','시간대_21~24_매출_비율'
         ]

csv1=pd.read_csv("서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2019년.csv",usecols=columns, encoding='euc-kr')
csv2=pd.read_csv("서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2020년.csv",usecols=columns, encoding='euc-kr')
csv3=pd.read_csv("서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2021년.csv",usecols=columns, encoding='euc-kr')
csvfile=pd.concat([csv1,csv2,csv3])
### 상권_구분_코드 ###

csvfile=csvfile.loc[csvfile['상권_코드_명'].isin(yongsan)]
print(csvfile.shape) #433551, 80 -> 14504, 53 
print(csvfile.shape[0])
print(csvfile.info())
# csvfile = csvfile.replace(0, np.NaN)
# print(csvfile.info())
#
# csvfile['주중_매출_금액']=csvfile['주중_매출_금액'].fillna(csvfile['주중_매출_금액'].mean())
# csvfile['주말_매출_금액']=csvfile['주말_매출_금액'].fillna(csvfile['주말_매출_금액'].mean())
# csvfile['월요일_매출_금액']=csvfile['월요일_매출_금액'].fillna(csvfile['월요일_매출_금액'].mean())
# csvfile['화요일_매출_금액']=csvfile['화요일_매출_금액'].fillna(csvfile['화요일_매출_금액'].mean())
# csvfile['수요일_매출_금액']=csvfile['수요일_매출_금액'].fillna(csvfile['수요일_매출_금액'].mean())
# csvfile['목요일_매출_금액']=csvfile['목요일_매출_금액'].fillna(csvfile['목요일_매출_금액'].mean())
# csvfile['금요일_매출_금액']=csvfile['금요일_매출_금액'].fillna(csvfile['금요일_매출_금액'].mean())
# csvfile['토요일_매출_금액']=csvfile['토요일_매출_금액'].fillna(csvfile['토요일_매출_금액'].mean())
# csvfile['일요일_매출_금액']=csvfile['일요일_매출_금액'].fillna(csvfile['일요일_매출_금액'].mean())
# csvfile['시간대_00~06_매출_금액']=csvfile['시간대_00~06_매출_금액'].fillna(csvfile['시간대_00~06_매출_금액'].mean())
# csvfile['시간대_06~11_매출_금액']=csvfile['시간대_06~11_매출_금액'].fillna(csvfile['시간대_06~11_매출_금액'].mean())
# csvfile['시간대_11~14_매출_금액']=csvfile['시간대_11~14_매출_금액'].fillna(csvfile['시간대_11~14_매출_금액'].mean())
# csvfile['시간대_14~17_매출_금액']=csvfile['시간대_14~17_매출_금액'].fillna(csvfile['시간대_14~17_매출_금액'].mean())
# csvfile['시간대_17~21_매출_금액']=csvfile['시간대_17~21_매출_금액'].fillna(csvfile['시간대_17~21_매출_금액'].mean())
# csvfile['시간대_21~24_매출_금액']=csvfile['시간대_21~24_매출_금액'].fillna(csvfile['시간대_21~24_매출_금액'].mean())
# csvfile['남성_매출_금액']=csvfile['남성_매출_금액'].fillna(csvfile['남성_매출_금액'].mean())
# csvfile['여성_매출_금액']=csvfile['여성_매출_금액'].fillna(csvfile['여성_매출_금액'].mean())
# csvfile['연령대_10_매출_금액']=csvfile['연령대_10_매출_금액'].fillna(csvfile['연령대_10_매출_금액'].mean())
# csvfile['연령대_20_매출_금액']=csvfile['연령대_20_매출_금액'].fillna(csvfile['연령대_20_매출_금액'].mean())
# csvfile['연령대_30_매출_금액']=csvfile['연령대_30_매출_금액'].fillna(csvfile['연령대_30_매출_금액'].mean())
# csvfile['연령대_40_매출_금액']=csvfile['연령대_40_매출_금액'].fillna(csvfile['연령대_40_매출_금액'].mean())
# csvfile['연령대_50_매출_금액']=csvfile['연령대_50_매출_금액'].fillna(csvfile['연령대_50_매출_금액'].mean())
# csvfile['연령대_60_이상_매출_금액']=csvfile['연령대_60_이상_매출_금액'].fillna(csvfile['연령대_60_이상_매출_금액'].mean())
# csvfile['주중_매출_건수']=csvfile['주중_매출_건수'].fillna(csvfile['주중_매출_건수'].mean())
# csvfile['주말_매출_건수']=csvfile['주말_매출_건수'].fillna(csvfile['주말_매출_건수'].mean())
# csvfile['월요일_매출_건수']=csvfile['월요일_매출_건수'].fillna(csvfile['월요일_매출_건수'].mean())
# csvfile['화요일_매출_건수']=csvfile['화요일_매출_건수'].fillna(csvfile['화요일_매출_건수'].mean())
# csvfile['수요일_매출_건수']=csvfile['수요일_매출_건수'].fillna(csvfile['수요일_매출_건수'].mean())
# csvfile['목요일_매출_건수']=csvfile['목요일_매출_건수'].fillna(csvfile['목요일_매출_건수'].mean())
# csvfile['금요일_매출_건수']=csvfile['금요일_매출_건수'].fillna(csvfile['금요일_매출_건수'].mean())
# csvfile['토요일_매출_건수']=csvfile['토요일_매출_건수'].fillna(csvfile['토요일_매출_건수'].mean())
# csvfile['일요일_매출_건수']=csvfile['일요일_매출_건수'].fillna(csvfile['일요일_매출_건수'].mean())
# csvfile['시간대_건수~06_매출_건수']=csvfile['시간대_건수~06_매출_건수'].fillna(csvfile['시간대_건수~06_매출_건수'].mean())
# csvfile['시간대_건수~11_매출_건수']=csvfile['시간대_건수~11_매출_건수'].fillna(csvfile['시간대_건수~11_매출_건수'].mean())
# csvfile['시간대_건수~14_매출_건수']=csvfile['시간대_건수~14_매출_건수'].fillna(csvfile['시간대_건수~14_매출_건수'].mean())
# csvfile['시간대_건수~17_매출_건수']=csvfile['시간대_건수~17_매출_건수'].fillna(csvfile['시간대_건수~17_매출_건수'].mean())
# csvfile['시간대_건수~21_매출_건수']=csvfile['시간대_건수~21_매출_건수'].fillna(csvfile['시간대_건수~21_매출_건수'].mean())
# csvfile['시간대_건수~24_매출_건수']=csvfile['시간대_건수~24_매출_건수'].fillna(csvfile['시간대_건수~24_매출_건수'].mean())
# csvfile['시간대_00~06_매출_비율']=csvfile['시간대_00~06_매출_비율'].fillna(csvfile['시간대_00~06_매출_비율'].mean())
# csvfile['시간대_06~11_매출_비율']=csvfile['시간대_06~11_매출_비율'].fillna(csvfile['시간대_06~11_매출_비율'].mean())
# csvfile['시간대_11~14_매출_비율']=csvfile['시간대_11~14_매출_비율'].fillna(csvfile['시간대_11~14_매출_비율'].mean())
# csvfile['시간대_14~17_매출_비율']=csvfile['시간대_17~21_매출_비율'].fillna(csvfile['시간대_21~24_매출_비율'].mean())
# csvfile['남성_매출_건수']=csvfile['남성_매출_건수'].fillna(csvfile['남성_매출_건수'].mean())
# csvfile['여성_매출_건수']=csvfile['여성_매출_건수'].fillna(csvfile['여성_매출_건수'].mean())
# csvfile['연령대_10_매출_건수']=csvfile['연령대_10_매출_건수'].fillna(csvfile['연령대_10_매출_건수'].mean())
# csvfile['연령대_20_매출_건수']=csvfile['연령대_20_매출_건수'].fillna(csvfile['연령대_20_매출_건수'].mean())
# csvfile['연령대_30_매출_건수']=csvfile['연령대_30_매출_건수'].fillna(csvfile['연령대_30_매출_건수'].mean())
# csvfile['연령대_40_매출_건수']=csvfile['연령대_40_매출_건수'].fillna(csvfile['연령대_40_매출_건수'].mean())
# csvfile['연령대_50_매출_건수']=csvfile['연령대_50_매출_건수'].fillna(csvfile['연령대_50_매출_건수'].mean())
# csvfile['연령대_60_이상_매출_건수']=csvfile['연령대_60_이상_매출_건수'].fillna(csvfile['연령대_60_이상_매출_건수'].mean())
#
# csvfile['주말_매출_금액']=sum(csvfile['토요일_매출_금액'],csvfile['일요일_매출_금액'])
# csvfile['주중_매출_금액']=csvfile['월요일_매출_금액']+csvfile['화요일_매출_금액']+csvfile['수요일_매출_금액']+csvfile['목요일_매출_금액']+csvfile['금요일_매출_금액']
# csvfile['분기당_매출_금액']=sum(csvfile['주말_매출_금액'],csvfile['주중_매출_금액'])

print(csvfile.shape) # 14504, 53
print(csvfile.info())
csvfile.columns=csvfile.columns.str.replace(r'~', r'_', regex=True)

csvfile.to_csv('yongsan2021.csv', encoding='utf-8')
print('용산구상권 파일 저장완료')