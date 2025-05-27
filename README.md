# 📝 Time Series Analysis: Stationarity & ARIMA

본 실습은 시계열 데이터의 정상성 분석부터 ARIMA 모델링, 예측 및 평가까지의 전체 흐름을 구현하며 학습한 기록입니다.

Melbourne 기온 데이터와 국제 항공 승객 수 데이터를 대상으로 시계열의 성질을 진단하고,  
비정상 시계열에 대해 ARIMA 모델을 적용하여 예측 성능을 평가하는 것을 목표로 합니다.

>📂 &nbsp;time_series_practice &nbsp;> &nbsp;time_series_arima_process.ipynb

<br>
<br>

## 실습 목적

- 시계열 데이터의 정상성과 비정상성 차이를 이해하고 시각적·통계적으로 판단한다.
- 로그 변환, 이동 평균 제거, 차분, 시계열 분해 등을 통해 비정상 시계열을 안정화한다.
- ACF/PACF 분석을 기반으로 ARIMA(p,d,q) 모델을 설정하고, 학습 및 예측을 수행한다.
- 예측 결과를 MAPE 등 지표로 정량 평가하여 모델 성능을 해석한다.

<br>
<br>

## 사용 데이터

1. Melbourne 기온 데이터 (`ts1`)
    - Daily Minimum Temperatures in Melbourne (1981–1990)

2. 국제 항공 승객 수 데이터 (`ts2`)
    - Monthly International Airline Passengers (1949–1960)

<br>
<br>

## 실습 프로세스

```
📂 1. 데이터 로드 및 시계열 구조화
│
├── ts1: 멜버른 기온 데이터
│     - CSV → datetime 인덱스
│     - 컬럼명: 'Temp'
│     - Series(ts1)로 추출
│
└── ts2: 국제 항공 승객 수 데이터
      - CSV → datetime 인덱스
      - 컬럼명: 'Passengers'
      - Series(ts2)로 추출


📂 2. 결측치 및 이상치 처리
│
├── ts1
│   └── '?0.2' 등 비정상값 → NaN → dropna()
│
└── ts2
    └── 결측 없음 → 그대로 사용


📂 3. 정성적 정상성 분석 (시각화)
│
├── plot_rolling_statistics(ts1)
│   └── Rolling Mean·Std 일정 → 정상성
│
└── plot_rolling_statistics(ts2)
    └── 추세와 분산 증가 → 비정상성 추정


📂 4. 정량적 정상성 분석 (ADF Test)
│
├── augmented_dickey_fuller_test(ts1)
│   └── p ≈ 0 → 정상성
│
└── augmented_dickey_fuller_test(ts2)
    └── p ≈ 1 → 비정상성


📂 5. 비정상 시계열(ts2) 변환 → 정상화
│
├── 로그 변환
│   └── ts_log = np.log(ts2)
│   └── augmented_dickey_fuller_test(ts2)
│       └── p ≈ 0.42 → 비정상성
│
├── 이동 평균 제거
│   └── moving_avg = ts_log.rolling(12).mean()
│   └── ts_log_moving_avg = ts_log - moving_av
│   └── ts_log_moving_avg.dropna()
│
├── ADF Test
│   └── plot_rolling_statistics(ts_log_moving_avg)
│   └── augmented_dickey_fuller_test(ts_log_moving_avg)
│       → p ≈ 0.02 → 거의 정상
│       → 추세 제거 시 window 크기 선택이 중요
│
└── 차분 수행
│   └── ts_log_moving_avg - ts_log_moving_avg.shift(-1)
│   └── dropna() 후 ADF Test
│       → p ≈ 0.0019 → 완전한 정상성 확보
│
└──  각 과정이 지니는 의미 이해


📂 6. 시계열 분해 (Decomposition)
│
├── decomposition = seasonal_decompose(ts_log)
│   ├── original
│   ├── trend
│   ├── seasonal
│   └── residual (✔ 분석 대상)
│
├── plot_rolling_statistics(residual)
├── augmented_dickey_fuller_test(residual)
│   └── p ≈ 0.000 → 매우 안정적인 정상 시계열
└──  ✔ residual을 기반으로 모델링 가능


📂 7. ACF / PACF 분석 → ARIMA 파라미터 추정
│
├── plot_acf(ts_log) → q 추정
│     ↳ 완만한 감소 → q = 0~1
│
├── plot_pacf(ts_log) → p 추정
│     ↳ lag 1, lag 14 → p = 1 또는 14
│
└── 차분 횟수(d) 추정
    ├── diff_1 = ts_log.diff(1) → p ≈ 0.07
    ├── diff_2 = diff_1.diff(1) → p ≈ 0
    └── d = 1 또는 2 시도 가능


📂 8. ARIMA 모델 훈련 및 예측
│
├── 데이터 분할
│   ├── train_data = ts_log[:90%]
│   └── test_data = ts_log[90%:]
│
├── 모델 학습
│   └── model = ARIMA(train_data, order=(14, 1, 0))
│   └── fitted_m = model.fit()
│
├── 훈련 예측 시각화
│   └── predict()  vs train_data
│
└── 테스트 예측
    ├── forecast(len(test_data), alpha=0.05) → fc
    ├── fc_series = pd.Series(fc, index=test_data.index)
    └── 시각화로 actual vs predicted 비교


📂 9. 예측 성능 평가
│
├── np.exp(test_data), np.exp(fc) → 원래 단위 복원
│
├── 평가 지표 계산
│   ├── MSE, MAE, RMSE
│   └── MAPE = 2.62%
│
└── 비교적 높은 정확도 / 모델 성능 양호

```

<br>

## 결론

항공 승객 수 데이터(ts2)는 비정상성을 띄고 있었고, 로그 변환과 차분을 통해 ADF p-value를 0.0019 이하로 낮추어 정상성을 확보하였다.

ACF, PACF 분석 결과를 바탕으로 ARIMA(14, 1, 0) 모델을 구성하여 학습 및 예측을 수행하였다.

테스트셋 예측 결과는 실제 시계열 흐름과 유사하게 나타났으며, 로그 역변환 후 예측 정확도는 다음과 같았다.

```
MSE   : 147.58  
MAE   : 9.32  
RMSE  : 12.14  
MAPE  : 2.62%
```

- PACF가 lag=14에서 유의미한 자기상관을 보여준다는 점이 실제 p=14 설정의 이론적 근거가 될 수 있었다.
- PACF에서 p=1을 해석한 모델보다 더 나은 성능으로, 실험적 튜닝이 유효했음을 보여준다.

최종적으로, 평균 예측 오차율 2.62% 수준의 비교적 높은 예측 정확도를 달성하였다.


<br>
<br>


## 인사이트 및 회고

이 실습을 통해 정상성과 예측 가능성이 가지는 의미, 그리고 모델링 전처리의 해석적 중요성을 경험적으로 이해할 수 있었다.

ARIMA 모델 학습에는 로그 변환된 ts_log 데이터만 사용했지만 그 이전의 추세 제거, 차분, 분해 과정은 모델 설계와 파라미터 추정의 근거를 확보하기 위한 탐색적 분석이 필요하다는 사실을 확인하였다.

정상성 확보는 로그 변환만으로 충분하지 않았고, 이동 평균 제거, 차분, ADF Test 등을 통해 적절한 차분 횟수(d)를 결정할 수 있었다.

모델 성능은 이론적 추정만으로 수행하는 것이 아니라 실험적으로 다양한 조합을 테스트하며 결정하는 것이 중요하다는 실험 결과 수치로 확인할 수 있었다.

<br>

>이 repository는 시계열 데이터 분석에서 정상성이 가지는 의미를 이해하고, 예측 모델링 과정에서 이를 어떻게 인지하고 반영하는가를 학습하는 데 중점을 두었습니다.
