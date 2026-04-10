# IMBK_Bank_Customer_Churn_ML

# 프로젝트 개요
- 배경: 은행 산업에서 신규 고객 유치보다 기존 고객 유지가 비용 효율적이며, 이를 위해 고객의 이탈(Churn)을 미리 예측하는 것이 중요
- 목표: Kaggle의 'Bank Customer Churn Dataset'을 활용하여 고객의 이탈 여부를 예측하는 머신러닝 모델을 개발하고, 주요 이탈 요인을 분석
- 기간: 2026년 4월 10일

# 사용 데이터
- 데이터 출처: Bank Customer Churn Dataset (row: 10000, col:12)
- 주요 변수: 국가(country), 성별(gender), 나이, 신용 점수, 잔고, 상품 보유 수, 활성 고객 여부 등

# 기술 스택
- 데이터 전처리: Pandas, Numpy, Scikit-learn(Preprocessing), Matplotlib, Seaborn
- 모델링: PyCaret, Scikit-learn(Ensemble), XGBoost, LightGBM, CatBoost
- 최적화: Optuna
- 사후분석: SHAP

# 데이터 전처리
1. 결측치 확인
2. 불필요한 변수 제거 - customer id
3. 범주형 변수 인코딩 - country, gender

# EDA
1. 칼럼 별 이탈 여부와 상관관계
<img width="1778" height="1180" alt="output" src="https://github.com/user-attachments/assets/5a9bf405-7777-4784-999f-374e22879e7f" />
- y(이탈 여부)의 비율 확인 결과, 이탈 고객에 비해 유지 고객의 비율이 훨씬 높다 -> 클래스 불균형
- 나이에 따른 이탈여부를 kde그래프로 확인 결과, 유지고객은 30대가 가장 높고, 이탈고객은 40대가 가장 많았다.
- 국가별 이탈률 확인 결과, 1위: 독일 / 2위: 스페인 / 3위: 프랑스 순으로 나왔다.
- 수 간 상관관계를 히트맵으로 확인한 결과, 큰 상관관계를 보이는 변수의 조합은 발견되지 않았고, 그나마 상품의 수와 계좌 잔액이 -0.30으로 음의 상관관계를 보이고 있다.
