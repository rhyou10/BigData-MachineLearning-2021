# -*- coding: utf-8 -*-
"""
Created on 2021

@author: Administrator
"""

#%%

# =============================================================================
# =============================================================================
# # 문제 01 유형(DataSet_01.csv 이용)
#
# 구분자 : comma(“,”), 4,572 Rows, 5 Columns, UTF-8 인코딩
# 
# 글로벌 전자제품 제조회사에서 효과적인 마케팅 방법을 찾기
# 위해서 채널별 마케팅 예산과 매출금액과의 관계를 분석하고자
# 한다.
# 컬 럼 / 정 의  /   Type
# TV   /     TV 마케팅 예산 (억원)  /   Double
# Radio / 라디오 마케팅 예산 (억원)  /   Double
# Social_Media / 소셜미디어 마케팅 예산 (억원)  / Double
# Influencer / 인플루언서 마케팅
# (인플루언서의 영향력 크기에 따라 Mega / Macro / Micro / 
# Nano) / String

# SALES / 매출액 / Double
# =============================================================================
# =============================================================================


# 실행시 블럭 잡고 F9
import pandas as pd

data01 = pd.read_csv('Dataset/Dataset_01.csv')



#%%

# =============================================================================
# 1. 데이터 세트 내에 총 결측값의 개수는 몇 개인가? (답안 예시) 23  26
# =============================================================================

data01.isna().sum().sum()
(data01.isna().sum(axis=1)>=1).sum()
data01.info()


#%%

# =============================================================================
# 2. TV, Radio, Social Media 등 세 가지 다른 마케팅 채널의 예산과 매출액과의 상관분석을
# 통하여 각 채널이 매출에 어느 정도 연관이 있는지 알아보고자 한다. 
# - 매출액과 가장 강한 상관관계를 가지고 있는 채널의 상관계수를 소수점 5번째
# 자리에서 반올림하여 소수점 넷째 자리까지 기술하시오. (답안 예시) 0.1234
# =============================================================================


data01.columns
# ['TV', 'Radio', 'Social_Media', 'Influencer', 'Sales'], dtype='object')
var =  ['TV', 'Radio', 'Social_Media', 'Sales']
q2 = data01[var].corr().abs().drop('Sales')['Sales']
q2
q2.max()
q2.argmax()
q2.idxmax()
q2.nlargest(1) # 상위값 개수 지정

q2.min()
q2.argmin()
q2.idxmin()
#q2.nsmalllest(1)
#0.9995



#%%

# =============================================================================
# 3. 매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을
# 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로
# 기술하시오. 
# - 분석 시 결측치가 포함된 행은 제거한 후 진행하며, 회귀계수는 소수점 넷째 자리
# 이하는 버리고 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================


data01 = data01.dropna()

x = data01[['TV', 'Radio', 'Social_Media']]
y = data01[['Sales']]



from statsmodels.formula.api import ols
from statsmodels.api import add_constant
from sklearn.linear_model import LinearRegression

var = ['TV', 'Radio', 'Social_Media']
lm = LinearRegression(fit_intercept=True).fit(data01[var], data01.Sales) # 절편제거(fit_inter...)
dir(lm)
lm.coef_
#[ 3.562,  0.004, -0.003]


data_x ='+'.join(x)
model = ols('Sales ~'+data_x, data01)
res = model.fit()
res.summary()
res.outlier_test() # 0.05 귀무가설 test
res.params # coef 값이지만 summary 보다 조금더 정확하게 나온다.
res.params.drop('Intercept').sort_values(ascending=False)# false가 내림차순

#help(res.params.drop('Intercept').sort_values) # 모를때 쳐볼것

(res.outlier_test()['unadj_p'] <0.05).sum()
data01[res.outlier_test()['unadj_p'] <0.05] #이상치 찾기

# ----
from statsmodels.api import OLS

xx=data01[var]
xx2 = add_constant(xx) #상수항추가

ols2 = OLS(data01.Sales , xx2)
result = ols2.fit()
result.summary()

#%%

# =============================================================================
# =============================================================================
# # 문제 02 유형(DataSet_02.csv 이용)
# 구분자 : comma(“,”), 200 Rows, 6 Columns, UTF-8 인코딩

# 환자의 상태와 그에 따라 처방된 약에 대한 정보를 분석하고자한다
# 
# 컬 럼 / 정 의  / Type
# Age  / 연령 / Integer
# Sex / 성별 / String
# BP / 혈압 레벨 / String
# Cholesterol / 콜레스테롤 레벨 /  String
# Na_to_k / 혈액 내 칼륨에 대비한 나트륨 비율 / Double
# Drug / Drug Type / String
# =============================================================================
# =============================================================================

data02 = pd.read_csv('Dataset/Dataset_02.csv')


#%%

# =============================================================================
# 1.해당 데이터에 대한 EDA를 수행하고, 여성으로 혈압이 High, Cholesterol이 Normal인
# 환자의 전체에 대비한 비율이 얼마인지 소수점 네 번째 자리에서 반올림하여 소수점 셋째
# 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================


len(data02[(data02.Sex=='F') & (data02.BP=='HIGH') & (data02.Cholesterol == 'NORMAL')]) / len(data02)
# 0.105
data02[data02.Sex=='F'][data02.BP=='HIGH'][data02.Cholesterol=='NORMAL']

q1 = pd.crosstab(index=[data02.Sex, data02.BP],
              columns=[data02.Cholesterol],
              normalize=True)
q1
q1.loc[('F','HIGH'), 'NORMAL']

data02[['Sex', 'BP', 'Cholesterol']].value_counts(normalize=True)

#%%

# =============================================================================
# 2. Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기
# 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오. 
# - Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은
# ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. 
# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 
# 초과는 ‘Lv4’로 변환하시오.
# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.
# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술하시오.
# (답안 예시) 3, 1.23456
# =============================================================================


# 1) 변수변경
q2 = data02.copy() # 복사본으로 하는게 좋다
q2['Age_gr'] = q2['Age'].apply(lambda x : 10 if x<20 else(20 if 20<x<30  else(30 if 30<x<40 else(40 if 40<x<50 else(50 if 50<x<60 else 60)))))
q2['Na_K_gr'] = q2['Na_to_K'].apply(lambda x : 'Lv1' if x <= 10 else('Lv2' if x<=20 else ('Lv3' if x<=30 else 'Lv4')))

import numpy as np
#np.where(조건, 참인경우 실행문, 거짓인경우 실행문)
q2['Age_gr'] = np.where(q2.Age<20, '10', 
         np.where(q2.Age<30, '20', 
            np.where(q2.Age<40, '30',
                     np.where(q2.Age<50, '40',
                              np.where(q2.Age<60, '50', '60')))))

q2['Na_K_gr'] = np.where(q2.Na_to_K <=10, 'Lv1',
                         np.where(q2.Na_to_K <=20, 'Lv2', 
                         np.where(q2.Na_to_K <=30 ,'lv3', 'Lv4')))
# 2) 빈도표 작성 - 입력값
temp = pd.crosstab(index= q2['Sex'], 
                   columns = q2['Drug'])
temp

# 3) 카이스퀘어 검정
from scipy.stats import chi2_contingency

# crtl + i  함수위에 커서두고 help창 뜬다. 이걸통해 return 값 확인가능
chi2 = chi2_contingency(temp) #교차표값

chi2
#(2.119248418109203, #chi2 통게량
# 0.7138369773987128, # p-value --> 두개의 변수가 독립이다.
# 4,   # 자유도, (r-1)(c-1)
# array([[43.68, 11.04,  7.68,  7.68, 25.92],
#        [47.32, 11.96,  8.32,  8.32, 28.08]]))

chi2[1]

q2_out = []
for col in ['Sex', 'BP', 'Cholesterol', 'Age_gr', 'Na_K_gr']:
    temp = pd.crosstab(index= q2[col], 
                   columns = q2['Drug'])
    chi2 = chi2_contingency(temp)
    if chi2[1] < 0.05:
        print(chi2[1])
        q2_out.append([col,chi2[0],chi2[1]])
        
q2_out = pd.DataFrame(q2_out,
                      columns=['x', 'chi', 'p_value'])
q2_out['p_value'].max()
# 4, 0.00070

#%%

# =============================================================================
# 3.Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한
# 분석을 수행하시오.
# - Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성
# - BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성
# - Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
# - Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를
# 수행하고 Root Node의 split feature와 split value를 기술하시오. 
# 이 때 split value는 소수점 셋째 자리까지 반올림하여 기술하시오. (답안 예시) Age, 
# 12.345
# =============================================================================



# 1) 더미변수 만들기
q3 =  data02.copy()

q3['Sex_cd'] = np.where(q3.Sex=='M', 0,1)
q3['BP_cd'] = np.where(q3.BP=='LOW', 1,
                       np.where(q3.BP=='NORMAL', 2,3))
q3['Ch_cd'] = np.where(q3.Cholesterol=='NORMAL', 0,1)

# 2) 의사결정나무 수행

x_list = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']
y_label = list(q3.Drug.unique())
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

dt = DecisionTreeClassifier().fit(q3[x_list], q3.Drug)

plot_tree(dt, max_depth=2,
          feature_names=x_list,
          class_names=y_label,
          precision=3,
          fontsize=8)

print(export_text(dt,max_depth=2,
            feature_names=x_list))

## 루트노드 Na_to_k, 14.829


# 3) Rootnode의 split feature / split_value












#%%

# =============================================================================
# =============================================================================
# # 문제 03 유형(DataSet_03.csv 이용)
# 
# 구분자 : comma(“,”), 5,001 Rows, 8 Columns, UTF-8 인코딩
# 안경 체인을 운영하고 있는 한 회사에서 고객 사진을 바탕으로 안경의 사이즈를
# 맞춤 제작하는 비즈니스를 기획하고 있다. 우선 데이터만으로 고객의 성별을
# 파악하는 것이 가능할 지를 연구하고자 한다.
#
# 컬 럼 / 정 의 / Type
# long_hair / 머리카락 길이 (0 – 길지 않은 경우 / 1 – 긴
# 경우) / Integer
# forehead_width_cm / 이마의 폭 (cm) / Double
# forehead_height_cm / 이마의 높이 (cm) / Double
# nose_wide / 코의 넓이 (0 – 넓지 않은 경우 / 1 – 넓은 경우) / Integer
# nose_long / 코의 길이 (0 – 길지 않은 경우 / 1 – 긴 경우) / Integer
# lips_thin / 입술이 얇은지 여부 0 – 얇지 않은 경우 / 1 –
# 얇은 경우) / Integer
# distance_nose_to_lip_long / 인중의 길이(0 – 인중이 짧은 경우 / 1 – 인중이
# 긴 경우) / Integer
# gender / 성별 (Female / Male) / String
# =============================================================================
# =============================================================================

data03 = pd.read_csv('Dataset/Dataset_03.csv')

#%%

# =============================================================================
# 1.이마의 폭(forehead_width_cm)과 높이(forehead_height_cm) 사이의
# 비율(forehead_ratio)에 대해서 평균으로부터 3 표준편차 밖의 경우를 이상치로
# 정의할 때, 이상치에 해당하는 데이터는 몇 개인가? (답안 예시) 10
# =============================================================================


q1 = data03.copy()

q1['forehead_ratio'] = q1.forehead_width_cm / q1.forehead_height_cm

xbar = q1.forehead_ratio.mean()
std = q1.forehead_ratio.std()

## 이상치 구간
UB = xbar + 3*std
LB = xbar - 3*std

## 이상치 검출
q1[(q1.forehead_ratio > UB) | (q1.forehead_ratio < LB)]

## 3개


#%%

# =============================================================================
# 2.성별에 따라 forehead_ratio 평균에 차이가 있는지 적절한 통계 검정을 수행하시오.
# - 검정은 이분산을 가정하고 수행한다.
# - 검정통계량의 추정치는 절대값을 취한 후 소수점 셋째 자리까지 반올림하여
# 기술하시오.
# - 신뢰수준 99%에서 양측 검정을 수행하고 결과는 귀무가설 기각의 경우 Y로, 그렇지
# 않을 경우 N으로 답하시오. (답안 예시) 1.234, Y
# =============================================================================



## 그룹의 수가 2개여서 T 검정, 아닐경우 아노바검정?

q1.gender.unique()

gr_A = q1[q1.gender=='Male']['forehead_ratio']
gr_B = q1[q1.gender=='Female']['forehead_ratio']


from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, bartlett
## 일표본, 독립인 이표본, 대응인 이표본
##바틀렛은 등분산검정 h0는 등분산, h1은 이분산

bart1 = bartlett(gr_A, gr_B)
bart1.pvalue

q2_out = ttest_ind(gr_A, gr_B, equal_var=False)
q2_out

#Out[168]: Ttest_indResult(statistic=2.9994984197511543, pvalue=0.0027186702390657176)

abs(q2_out.statistic)
round(abs(q2_out.statistic),3)

#2.999, Y


#%%

# =============================================================================
# 3.주어진 데이터를 사용하여 성별을 구분할 수 있는지 로지스틱 회귀분석을 적용하여
# 알아 보고자 한다. 
# - 데이터를 7대 3으로 나누어 각각 Train과 Test set로 사용한다. 이 때 seed는 123으로
# 한다.
# - 원 데이터에 있는 7개의 변수만 Feature로 사용하고 gender를 label로 사용한다.
# (forehead_ratio는 사용하지 않음)
# - 로지스틱 회귀분석 예측 함수와 Test dataset를 사용하여 예측을 수행하고 정확도를
# 평가한다. 이 때 임계값은 0.5를 사용한다. 
# - Male의 Precision 값을 소수점 둘째 자리까지 반올림하여 기술하시오. (답안 예시) 
# 0.12
# 
# 
# (참고) 
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# train_test_split 의 random_state = 123
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
random_state = 123

q3 = data03.copy()
y = q3.gender
x = q3.drop(columns='gender')
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123)

lr = LogisticRegression()
lr.fit(x_train, y_train)

q3_out_class = lr.predict(x_test)
q3_out_class

q3_out_pr = lr.predict_proba(x_test)
q3_out_pr ## 확률

from sklearn.metrics import precision_score, classification_report

precision_score(y_test, q3_out_class, pos_label='Male')

#0.9588414634146342


print(classification_report(y_test, q3_out_class))


#%%

# =============================================================================
# =============================================================================
# # 문제 04 유형(DataSet_04.csv 이용)
#
#구분자 : comma(“,”), 6,718 Rows, 4 Columns, UTF-8 인코딩

# 한국인의 식생활 변화가 건강에 미치는 영향을 분석하기에 앞서 육류
# 소비량에 대한 분석을 하려고 한다. 확보한 데이터는 세계 각국의 1인당
# 육류 소비량 데이터로 아래와 같은 내용을 담고 있다.

# 컬 럼 / 정 의 / Type
# LOCATION / 국가명 / String
# SUBJECT / 육류 종류 (BEEF / PIG / POULTRY / SHEEP) / String
# TIME / 연도 (1990 ~ 2026) / Integer
# Value / 1인당 육류 소비량 (KG) / Double
# =============================================================================
# =============================================================================

# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# from scipy.stats import ttest_rel
# #3
# from sklearn.linear_model import LinearRegression

#%%
data04 = pd.read_csv('Dataset/Dataset_04.csv')


# =============================================================================
# 1.한국인의 1인당 육류 소비량이 해가 갈수록 증가하는 것으로 보여 상관분석을 통하여
# 확인하려고 한다. 
# - 데이터 파일로부터 한국 데이터만 추출한다. 한국은 KOR로 표기되어 있다.
# - 년도별 육류 소비량 합계를 구하여 TIME과 Value간의 상관분석을 수행하고
# 상관계수를 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지만 기술하시오. 
# (답안 예시) 0.55
# =============================================================================

q1 = data04.copy()
q1 = q1[q1.LOCATION == 'KOR']

## pivot
q1_tab = pd.pivot_table(q1, index = 'TIME',
                        values= 'Value',
                        aggfunc='sum').reset_index()

## gropuby
x = q1['Value'].groupby(q1.TIME).sum()
x = pd.DataFrame(x)
x.reset_index(inplace=True)
x.corr()

# 0.96


#%%

# =============================================================================
# 2. 한국 인근 국가 가운데 식생의 유사성이 상대적으로 높은 일본(JPN)과 비교하여, 연도별
# 소비량에 평균 차이가 있는지 분석하고자 한다.
# - 두 국가의 육류별 소비량을 연도기준으로 비교하는 대응표본 t 검정을 수행하시오.
# - 두 국가 간의 연도별 소비량 차이가 없는 것으로 판단할 수 있는 육류 종류를 모두
# 적으시오. (알파벳 순서) (답안 예시) BEEF, PIG, POULTRY, SHEEP
# =============================================================================


q2 = data04[data04.LOCATION.isin(['KOR','JPN'])]

## 육류 종류, 연도별로 대응이되도록 데이터 생성
sub_list = q2.SUBJECT.unique()


## 대등 T-test
from scipy.stats import ttest_rel## 데이터가 쌍으로 들어가있어야만 가능하다.

q2_out = []
for sub in sub_list:
    temp = q2[q2.SUBJECT == sub]
    q2_tab = pd.pivot_table(temp, index='TIME',
                   columns='LOCATION',
                   values='Value').dropna() #결측치제거
    ttest_out = ttest_rel(q2_tab.JPN, q2_tab.KOR)
    pvalue=ttest_out.pvalue
    q2_out.append([sub,pvalue])
    
q2_out = pd.DataFrame(q2_out, columns=['sub', 'p_val'])    
q2_out[q2_out.p_val>=0.05]   

#POULTRY
    
#%%

# =============================================================================
# 3.(한국만 포함한 데이터에서) Time을 독립변수로, Value를 종속변수로 하여 육류
# 종류(SUBJECT) 별로 회귀분석을 수행하였을 때, 가장 높은 결정계수를 가진 모델의
# 학습오차 중 MAPE를 반올림하여 소수점 둘째 자리까지 기술하시오. (답안 예시) 21.12
# (MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
# (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
# 
# =============================================================================

## 한국데이터 출력
q3 = q1.copy()
sub_list = q3.SUBJECT.unique()


#육류 종류별로 회귀분석 : 결정계수, mape(예측값구해야함) 도 구해야한다.
from sklearn.linear_model import LinearRegression

## 종류별로 하라고 했으므로 따로따로 돌려야한다.
temp = q3[q3.SUBJECT == 'BEEF']
#lm = LinearRegression().fit(temp.TIME, temp.Value)## 1차 구조로 안된다

## shape 바꾸기 [] 추가
temp.TIME.shape #(36,) 
temp.TIME.values.reshape(-1,1).shape
temp[['TIME']].shape

lm = LinearRegression().fit(temp[['TIME']], temp.Value)
pred = lm.predict(temp[['TIME']])
r2_score = lm.score(temp[['TIME']], temp.Value)


q3_out = []
for sub in sub_list:
    temp = q3[q3.SUBJECT == sub]
    lm = LinearRegression().fit(temp[['TIME']], temp.Value)
    pred = lm.predict(temp[['TIME']])
    r2_score = lm.score(temp[['TIME']], temp.Value)
    mape = (abs(temp['Value'] - pred)/temp['Value']).sum()*100/len(temp)
    q3_out.append([sub,r2_score, mape]) 

q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2', 'mape'])

q3_out.r2.max()
q3_out.r2.idxmax()
q3_out.mape[q3_out.r2.idxmax()]

q3_out.sort_values(by='r2',ascending=False)

# 5.78


##번외

#globals()['lm_'+sub] 반복적으로 돌때마다 변수이름 새롭게 생성가능하다.

for sub in sub_list:
    temp = q3[q3.SUBJECT == sub]
    globals()['lm_'+sub]= LinearRegression().fit(temp[['TIME']], temp.Value)
    pred = eval('lm_'+sub).predict(temp[['TIME']])
    r2_score = eval('lm_'+sub).score(temp[['TIME']], temp.Value)
    mape = (abs(temp['Value'] - pred)/temp['Value']).sum()*100/len(temp)
    q3_out.append([sub,r2_score, mape]) 


#%%

# =============================================================================
# =============================================================================
# # 문제 05 유형(DataSet_05.csv 이용)
#
# 구분자 : comma(“,”), 8,068 Rows, 12 Columns, UTF-8 인코딩
#
# A자동차 회사는 신규 진입하는 시장에 기존 모델을 판매하기 위한 마케팅 전략을 
# 세우려고 한다. 기존 시장과 고객 특성이 유사하다는 전제 하에 기존 고객을 세분화하여
# 각 그룹의 특징을 파악하고, 이를 이용하여 신규 진입 시장의 마케팅 계획을 
# 수립하고자 한다. 다음은 기존 시장 고객에 대한 데이터이다.
#

# 컬 럼 / 정 의 / Type
# ID / 고유 식별자 / Double
# Age / 나이 / Double
# Age_gr / 나이 그룹 (10/20/30/40/50/60/70) / Double
# Gender / 성별 (여성 : 0 / 남성 : 1) / Double
# Work_Experience / 취업 연수 (0 ~ 14) / Double
# Family_Size / 가족 규모 (1 ~ 9) / Double
# Ever_Married / 결혼 여부 (Unknown : 0 / No : 1 / Yes : 2) / Double
# Graduated / 재학 중인지 여부 / Double
# Profession / 직업 (Unknown : 0 / Artist ~ Marketing 등 9개) / Double
# Spending_Score / 소비 점수 (Average : 0 / High : 1 / Low : 2) / Double
# Var_1 / 내용이 알려지지 않은 고객 분류 코드 (0 ~ 7) / Double
# Segmentation / 고객 세분화 결과 (A ~ D) / String
# =============================================================================
# =============================================================================


#(참고)
#1
# import pandas as pd
# #2
# from scipy.stats import chi2_contingency
# #3
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# import pydot


#%%

# =============================================================================
# 1.위의 표에 표시된 데이터 타입에 맞도록 전처리를 수행하였을 때, 데이터 파일 내에
# 존재하는 결측값은 모두 몇 개인가? 숫자형 데이터와 문자열 데이터의 결측값을
# 모두 더하여 답하시오.
# (String 타입 변수의 경우 White Space(Blank)를 결측으로 처리한다) (답안 예시) 123
# =============================================================================






#%%

# =============================================================================
# 2.이어지는 분석을 위해 결측값을 모두 삭제한다. 그리고, 성별이 세분화(Segmentation)에
# 영향을 미치는지 독립성 검정을 수행한다. 수행 결과, p-value를 반올림하여 소수점
# 넷째 자리까지 쓰고, 귀무가설을 기각하면 Y로, 기각할 수 없으면 N으로 기술하시오. 
# (답안 예시) 0.2345, N
# =============================================================================





#%%

# =============================================================================
# 3.Segmentation 값이 A 또는 D인 데이터만 사용하여 의사결정 나무 기법으로 분류
# 정확도를
# 측정해 본다. 
# - 결측치가 포함된 행은 제거한 후 진행하시오.
# - Train대 Test 7대3으로 데이터를 분리한다. (Seed = 123)
# - Train 데이터를 사용하여 의사결정나무 학습을 수행하고, Test 데이터로 평가를
# 수행한다.
# - 의사결정나무 학습 시, 다음과 같이 설정하시오:
# • Feature: Age_gr, Gender, Work_Experience, Family_Size, 
#             Ever_Married, Graduated, Spending_Score
# • Label : Segmentation
# • Parameter : Gini / Max Depth = 7 / Seed = 123
# 이 때 전체 정확도(Accuracy)를 소수점 셋째 자리 이하는 버리고 소수점 둘째자리까지
# 기술하시오.
# (답안 예시) 0.12
# =============================================================================



