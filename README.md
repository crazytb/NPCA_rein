# Semi-MDP NPCA Deep Reinforcement Learning

이 프로젝트는 Semi-MDP 기반 DQN을 사용하여 NPCA (Non-Primary Channel Access) 의사결정을 학습하는 시스템입니다.

## 🚀 빠른 시작

### 1. DRL 모델 훈련
```bash
# 기본 설정 (OBSS Duration: 100 slots)으로 훈련
python main_semi_mdp_training.py

# 특정 OBSS Duration으로 훈련 (예: 150 slots)
python main_semi_mdp_training.py 150
```

### 2. 모델 성능 비교
```bash
# 기본 설정으로 비교 (OBSS Duration: 100 slots)
python comparison_test.py

# 특정 OBSS Duration으로 비교 (예: 150 slots)
python comparison_test.py 150
```

## 📁 파일 구조

### 핵심 파일들
- `main_semi_mdp_training.py` - DRL 모델 훈련 스크립트
- `comparison_test.py` - 훈련된 모델 vs 베이스라인 비교
- `drl_framework/` - 핵심 DRL 프레임워크
  - `train.py` - Semi-MDP 학습 알고리즘
  - `network.py` - DQN 신경망
  - `random_access.py` - CSMA/CA 시뮬레이션
  - `configs.py` - 통합 설정 파일
  - `params.py` - 하이퍼파라미터
- `npca_semi_mdp_env.py` - Gymnasium 환경

### 실험 파일들
- `experimental_files/` - 모든 실험 및 분석 스크립트들

## 🎯 동작 원리

### DRL Agent 학습
1. **상황**: Primary 채널에서 CSMA/CA 백오프 중 OBSS 감지
2. **액션**: 
   - Action 0: StayPrimary (현재 채널에서 대기)
   - Action 1: GoNPCA (Secondary 채널로 이동)
3. **보상**: 
   - 성공적 전송: +33 (PPDU duration)
   - 전송 시도 비용: -5 (에너지 비용)

### 비교 베이스라인
- **Primary-Only**: 항상 Primary 채널에서 대기
- **NPCA-Only**: 항상 Secondary 채널로 이동
- **Random**: 무작위 선택
- **DRL**: 학습된 정책

## 📊 결과 출력

### 훈련 후 생성되는 파일들
- `./obss_comparison_results/trained_model_obss_X/model.pth` - 훈련된 모델
- `./obss_comparison_results/trained_model_obss_X/training_results.png` - 훈련 곡선

### 비교 후 생성되는 파일들
- `./comparison_results/policy_comparison.png` - 종합 비교 시각화
- `./comparison_results/comparison_results.csv` - 비교 결과 데이터

## ⚙️ 설정

모든 핵심 파라미터는 `drl_framework/configs.py`에서 통합 관리됩니다:

```python
# 핵심 설정
PPDU_DURATION = 33              # 프레임 전송 시간
ENERGY_COST = 5.0               # 전송 시도 에너지 비용
DEFAULT_NUM_EPISODES = 5000     # 훈련 에피소드 수
DEFAULT_NUM_SLOTS_PER_EPISODE = 200  # 에피소드당 슬롯 수
```

## 🔧 의존성

```bash
pip install torch pandas matplotlib gymnasium
```

## 📈 성능 메트릭

- **Average Reward**: 에피소드 평균 보상
- **Efficiency**: 채널 이용 효율성
- **Action Distribution**: 액션 선택 분포
- **Throughput**: 성공적 전송량

## 🎪 예제 워크플로우

```bash
# 1단계: 150 슬롯 OBSS duration으로 모델 훈련
python main_semi_mdp_training.py 150

# 2단계: 훈련된 모델 성능 비교
python comparison_test.py 150

# 결과: ./comparison_results/policy_comparison.png 확인
```

이제 간단하고 깔끔한 **훈련 → 비교** 워크플로우가 완성되었습니다!