# 비트코인 자동매매 프로그램 디렉토리 구조

현재 구현된 프로젝트의 디렉토리 구조는 다음과 같습니다:

```
bitcoin-trading-bot/
├── config/
│   ├── settings.py         # 설정 파일 및 환경 변수
│   └── .env                # 환경 변수 파일 (API 키 등)
│
├── data/
│   ├── collectors.py       # 업비트 API를 통한 데이터 수집
│   ├── indicators.py       # 기술적 지표 계산 (RSI, MACD, 볼린저 밴드 등)
│   ├── processors.py       # 데이터 전처리 및 가공 (노이즈 추가, 분할 등)
│   └── storage.py          # 데이터 저장 및 관리 (CSV, JSON 등)
│
├── models/
│   ├── lstm.py             # LSTM 딥러닝 모델 구현
│   ├── random_forest.py    # 랜덤 포레스트 모델 구현
│   ├── reinforcement.py    # PPO 강화학습 모델 구현
│   ├── ensemble.py         # 앙상블 모델 통합
│   └── gpt_analyzer.py     # GPT 기반 시장 분석
│
├── strategies/
│   ├── base.py             # 기본 전략 추상 클래스
│   ├── moving_average.py   # 이동평균 교차 전략
│   ├── rsi_strategy.py     # RSI 기반 매매 전략
│   └── trend_following.py  # 트렌드 팔로잉 전략
│
├── risk/
│   ├── position_sizing.py  # 포지션 크기 관리
│   └── stop_loss.py        # 손절매 전략 관리
│
├── backtest/
│   └── engine.py           # 백테스트 엔진 구현
│
├── utils/
│   ├── evaluation.py       # 성과 평가 지표 계산
│   ├── logging.py          # 로깅 시스템 구현
│   ├── monitoring.py       # 시스템 상태 모니터링
│   ├── recovery.py         # 시스템 복구 및 안전 종료
│   └── visualization.py    # 데이터 및 결과 시각화
│
└── main.py                 # 메인 실행 파일
```

## 구현된 주요 기능

### 데이터 관리
- 업비트 API 연동을 통한 데이터 수집
- 기술적 지표 계산 (RSI, MACD, 볼린저 밴드, EMA, ATR 등)
- 데이터 전처리 및 노이즈 추가
- CSV 및 JSON 형식 데이터 저장/로드

### 모델 구현
- LSTM 기반 가격 예측 및 방향성 예측
- 랜덤 포레스트 기반 방향성 및 시장 상태 분류
- PPO 알고리즘 기반 강화학습 리스크 관리
- 다양한 모델의 앙상블 구현
- GPT 기반 시장 분석 및 전략 추천

### 트레이딩 전략
- 기본 전략 추상 클래스
- 이동평균 교차 전략
- RSI 기반 매매 전략
- 트렌드 팔로잉 전략

### 리스크 관리
- 시장 상태에 따른 포지션 크기 조정
- 다양한 손절매 전략 구현

### 백테스트
- 백테스트 엔진 구현
- 다양한 성과 지표 계산

### 유틸리티 기능
1. **logging.py**: 일관된 로깅 설정 및 관리 기능
   - 모든 모듈에서 사용할 표준 로거 생성
   - 로그 레벨 설정 기능
   - 로그 파일 관리 (일자별, 모듈별 로그 파일 생성)

2. **monitoring.py**: 시스템 상태 모니터링 기능
   - 메모리 사용량, CPU 사용량 모니터링
   - API 호출 횟수 및 제한 모니터링
   - 거래 실행 상태 모니터링
   - 실시간 알림 기능

3. **evaluation.py**: 성과 지표 계산 기능
   - 수익률, 샤프 비율, 최대 낙폭 등 주요 지표 계산
   - 일일/주간/월간 성과 추적
   - 전략별 성과 비교 기능

4. **visualization.py**: 시각화 기능
   - 수익률 차트, 성과 차트 생성
   - 기술적 지표 시각화
   - 실시간 대시보드용 차트 생성
   - 백테스트 결과 시각화

5. **recovery.py**: 시스템 복구 및 안전 종료 기능
   - 현재 상태 저장 기능
   - 예기치 않은 셧다운 감지
   - 이전 상태에서 안전하게 복구하는 기능
   - 거래 중 오류 발생 시 안전 종료 절차

