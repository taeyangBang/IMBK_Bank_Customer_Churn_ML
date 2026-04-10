# IMBK_Bank_Customer_Churn_ML

# 프로젝트 개요
- 배경: 은행 산업에서 신규 고객 유치보다 기존 고객 유지가 비용 효율적이며, 이를 위해 고객의 이탈을 미리 예측하는 것이 중요
- 목표: Kaggle의 'Bank Customer Churn Dataset'을 활용하여 고객의 이탈 여부를 예측하는 머신러닝 모델을 개발하고, 주요 이탈 요인을 분석
- 기간: 2026년 4월 10일

# 사용 데이터
- 데이터 출처: Bank Customer Churn Dataset (row: 10000, col:12)
- 주요 변수: 국가, 성별, 나이, 신용 점수, 잔고, 상품 보유 수, 활성 고객 여부 등

# 기술 스택
- 데이터 전처리: Pandas, Numpy, Scikit-learn(Preprocessing), Matplotlib, Seaborn
- 모델링: PyCaret, Scikit-learn(Ensemble), XGBoost, LightGBM, CatBoost
- 최적화: Optuna
- 사후분석: SHAP

# 데이터 전처리
1. 결측치 확인 - 결측치 없음
2. 불필요한 변수 제거 - customer id
3. 범주형 변수 인코딩 - country, gender

# EDA
1. 칼럼 별 이탈 여부와 상관관계
<img width="1778" height="1180" alt="output" src="https://github.com/user-attachments/assets/5a9bf405-7777-4784-999f-374e22879e7f" />

- y(이탈 여부)의 비율 확인 결과, 이탈 고객에 비해 유지 고객의 비율이 훨씬 높다. -> 클래스 불균형
- 나이에 따른 이탈여부를 kde그래프로 확인 결과, 유지고객은 30대가 가장 높고, 이탈고객은 40대가 가장 많았다.
- 국가별 이탈률 확인 결과, 1위: 독일 / 2위: 스페인 / 3위: 프랑스 
- 변수 간 상관관계를 히트맵으로 확인한 결과, 뚜렷한 상관관계를 보이는 변수의 조합은 발견되지 않았고, 상품의 수와 계좌 잔액이 -0.30으로 음의 상관관계를 보이고 있다.

2. 상품 수별 고객 수 및 이탈률(%)
<img width="1380" height="580" alt="output2" src="https://github.com/user-attachments/assets/1a1a4006-f91d-4080-a68e-20f690ffc4fc" />

- 상품 수별 고객 수 및 이탈률 확인 결과, 상품 보유량이 많은 고객군에서 높은 이탈률이 나타났습니다. 다만, 이탈 클래스의 데이터 비중이 현저히 낮은 클래스 불균형 문제가 존재합니다. 분석의 신뢰도를 확보하기 위해 오버샘플링(SMOTE 등)이나 가중치 조정 등의 필요해 보인다.

# 모델링
1. AutoML - 상위 성능 모델 선정
<img width="982" height="557" alt="image" src="https://github.com/user-attachments/assets/a561a801-aa6f-4a87-b967-60bbc75638ef" />

- F1_score를 기준으로 상위 모델 선정. catboost, lightgbm, gbc, xgboost


2. optuna - 해당 모델 하이퍼파라미터 튜닝

| Model | Hyperparameter | Range |
| :--- | :----: | ---: |
| CatBoost | iterations | 100 ~ 500 |
|  | depth | 4 ~ 10 |
|  | learning_rate | 0.01 ~ 0.1 |
|  | l2_leaf_reg | 1 ~ 10 |
| LightGBM | n_estimators | 100 ~ 500 |
|  | max_depth | 3 ~ 15 |
|  | learning_rate | 0.01 ~ 0.1 |
|  | num_leaves | 20 ~ 150 |
|  | min_child_samples | 5 ~ 50 |
| XGBoost | n_estimators | 100 ~ 500 |
|  | max_depth | 3 ~ 15 |
|  | learning_rate | 0.01 ~ 0.1 |
|  | subsample | 0.6 ~ 1.0 |
|  | colsample_bytree | 0.6 ~ 1.0 |
| Gradient Boosting | n_estimators | 100 ~ 300 |
|  | learning_rate | 0.01 ~ 0.1 |
|  | max_depth | 3 ~ 10 |
|  | subsample | 0.6 ~ 1.0 |

3. 모델 학습 결과 - f1_score

| Model | f1_score |
| :--- | :----: |
| CatBoost | 0.6097560975609756 |
| LightGBM | 0.5789473684210527 |
| XGBoost | 0.593607305936073 |
| Gradient Boosting | 0.6033182503770739 |

4. Stacking
- 1차 모델: LightGBM, XGBoost, Gradient Boosting
- 메타 모델: CatBoost
- f1_score: 0.5824345146379045

# 사후 분석

<img width="758" height="546" alt="SHAP Value" src="https://github.com/user-attachments/assets/cb13d1f4-7223-4b01-b1e3-61d3557f1fd7" />


1. 주요 칼럼 해석

- products_number: 보유 중인 상품 수가 너무 많으면 이탈위험이 있다. 1~2개 정도 적당히 보유한 고객들이 유지 될 확률이 높다.
- age: 나이가 많을 수록 이탈 확률이 매우 높아지고, 젊은 세대는 상대적으로 이탈 확률이 낮다. 고령층 고객을 대상으로하는 프로모션 및 서비스가 필요해보인다.
- active_member: 활성 고객과 비활성 고객은 명확하게 갈라져있다. 활동이 없는 고객은 이탈할 가능성이 매우 높다. 비활성 고객이 되기전에 예측하여 여러 프로모션을 통해 고객유치전략을 세워야한다.
- balance: 잔고가 높을 수록 이탈률이 높은 경향이 있다. 우량 고객을 잡기 위한 전략이 필요하다.


2. 인사이트 제안 - 고객 이탈 방어

- 상품 다량 보유 고객 이탈 - 보유 상품의 수가 1개나 4개나 혜택의 차이가 크지 않을 수 있다. 상품의 수에 비례한 누적형 혜택 설계가 필요하다.
- 고령층 고객 이탈 - 디지털 기술이 불편한 고령층을 대상으로한 편의 서비스가 필요하다.
- 비활성 고객 이탈 - 단순히 활성, 비활성으로 나누는 것이 아닌 비활성 위험 고객을 사전에 확인하여 여러 프로모션을 통해 계속해서 이용률을 유지시켜야 한다.
- 우량 고객 이탈 - 고액 잔액 고객을 대상으로 고객전담서비스를 통해 이탈을 막아야한다. 





