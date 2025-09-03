# Semi-MDP 기반 NPCA STA 학습

이 프로젝트는 Semi-MDP를 사용하여 STA(Station)의 NPCA(Non-Primary Channel Access) 결정을 학습시키는 시스템입니다.

## 개요

STA가 Primary 채널이 OBSS(Overlapping BSS)로 점유된 상황에서 다음 두 액션 중 하나를 선택하도록 학습합니다:
- **Action 0**: StayPrimary (Primary 채널에서 대기)
- **Action 1**: GoNPCA (NPCA 채널로 이동하여 전송)

## 주요 구성 요소

### 1. 핵심 클래스
- `SemiMDPLearner`: DQN 기반 Semi-MDP 학습 알고리즘
- `STA`: 학습 가능한 Station 클래스 (learner 속성 포함)
- `Channel`: 채널 상태 관리 (intra-BSS, OBSS 점유)
- `Simulator`: 시뮬레이션 환경

### 2. 파일 구조
```
drl_framework/
├── train.py           # SemiMDPLearner 클래스 및 학습 함수
├── random_access.py   # STA, Channel, Simulator 클래스
├── network.py         # DQN 네트워크 및 ReplayMemory
├── params.py          # 하이퍼파라미터 설정
└── configs.py         # 시뮬레이션 설정

main_semi_mdp_training.py  # Semi-MDP 학습 실행 파일
main_npca_simulation.py    # 기존 시뮬레이션 실행 파일 (학습 없음)
```

## 사용법

### 1. 환경 설치
```bash
pip install torch pandas matplotlib
```

### 2. Semi-MDP 학습 실행
```bash
python main_semi_mdp_training.py
```

### 3. 학습 결과
- 모델: `./semi_mdp_results/semi_mdp_model.pth`
- 그래프: `./semi_mdp_results/training_results.png`

## Semi-MDP 구조

### 상태 관측 (State)
- `primary_channel_obss_occupied_remained`: Primary 채널의 OBSS 점유 남은 시간
- `radio_transition_time`: 라디오 전환 시간
- `tx_duration`: 전송 지속 시간
- `cw_index`: Contention Window 인덱스

### 액션 (Action)
- `0`: StayPrimary - Primary 채널에서 대기
- `1`: GoNPCA - NPCA 채널로 이동

### 보상 (Reward)
- 전송 완료 시: 성공한 PPDU 길이만큼 양의 보상
- 전송 실패 시: 보상 없음

### 옵션 (Option)
- 결정 시점: Primary 채널이 OBSS로 점유될 때
- 옵션 종료: 전송 완료 또는 Primary로 복귀할 때
- Semi-MDP 학습: γ^τ 할인을 적용한 누적 보상 학습

## 학습 파라미터

주요 하이퍼파라미터는 `drl_framework/params.py`에서 설정:
- `BATCH_SIZE = 128`: 배치 크기
- `GAMMA = 0.99`: 할인 인수
- `EPS_START = 0.9`: 초기 epsilon
- `EPS_END = 0.05`: 최종 epsilon
- `EPS_DECAY = 1000`: epsilon 감소율
- `TAU = 0.005`: Target network 업데이트율
- `LR = 1e-4`: 학습률

## 학습 과정 모니터링

학습 중 10 에피소드마다 출력되는 정보:
- `Avg Reward`: 최근 10 에피소드 평균 보상
- `Avg Loss`: 평균 손실
- `Epsilon`: 현재 탐험율
- `Memory Size`: Replay buffer 크기

## 확장 및 커스터마이징

### 1. 보상 함수 수정
`STA.calculate_reward()` 메서드를 수정하여 다른 보상 구조 적용 가능

### 2. 상태 관측 추가
`STA.get_obs()` 메서드에 새로운 관측값 추가 가능

### 3. 네트워크 구조 변경
`drl_framework/network.py`의 `DQN` 클래스 수정

### 4. 학습 설정 변경
`main_semi_mdp_training.py`의 `create_training_config()` 함수 수정

## 문제 해결

### Import 오류
- PyTorch, pandas, matplotlib 패키지 설치 확인
- Python path 설정 확인

### CUDA 오류
- CPU 모드로 강제 실행: `device = torch.device("cpu")`

### 메모리 부족
- `BATCH_SIZE` 감소
- `num_slots_per_episode` 감소
- Replay buffer 크기 감소

## 성능 평가

학습된 모델의 성능은 다음으로 평가:
1. 에피소드별 누적 보상 추이
2. 손실 함수 수렴성
3. NPCA 채널 활용률
4. 전송 성공률