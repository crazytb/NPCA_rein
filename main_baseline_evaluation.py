#!/usr/bin/env python3
"""
Baseline Evaluation Script for NPCA STA Policies

정확한 baseline 비교를 위해 main_semi_mdp_training.py와 동일한 환경에서
Primary-Only, NPCA-Only 정책의 성능을 평가합니다.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.configs import (
    PPDU_DURATION,
    PPDU_DURATION_VARIANTS,
    RADIO_TRANSITION_TIME,
    OBSS_GENERATION_RATE,
    DEFAULT_NUM_STAS_CH0,
    DEFAULT_NUM_STAS_CH1,
)

def create_baseline_config(obss_duration=None, ppdu_variant='medium', random_ppdu=False):
    """training과 동일한 설정 생성"""
    obss_duration = obss_duration or 100
    ppdu_duration = PPDU_DURATION_VARIANTS.get(ppdu_variant, PPDU_DURATION)

    channels = [
        Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE['secondary']),
        Channel(channel_id=1, obss_generation_rate=OBSS_GENERATION_RATE['primary'],
                obss_duration_range=(obss_duration, obss_duration))
    ]

    stas_config = []

    # CH0 STA들 (NPCA 비활성화)
    for i in range(DEFAULT_NUM_STAS_CH0):
        stas_config.append({
            "sta_id": i,
            "channel_id": 0,
            "npca_enabled": False,
            "ppdu_duration": ppdu_duration,
            "radio_transition_time": RADIO_TRANSITION_TIME
        })

    # CH1 STA들 (NPCA 활성화)
    for i in range(DEFAULT_NUM_STAS_CH1):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": ppdu_duration,
            "radio_transition_time": RADIO_TRANSITION_TIME
        })

    return channels, stas_config

class FixedPolicy:
    """고정 정책 클래스"""

    @staticmethod
    def primary_only():
        """항상 Primary 채널에 머무름"""
        return 0

    @staticmethod
    def npca_only():
        """항상 NPCA 채널로 이동"""
        return 1

    @staticmethod
    def random():
        """랜덤하게 선택"""
        return np.random.choice([0, 1])

def evaluate_baseline_policy(policy_func, policy_name, channels, stas_config,
                           num_episodes=100, num_slots_per_episode=11111, random_ppdu=False):
    """baseline 정책 평가 (training과 동일한 환경)"""

    print(f"Evaluating {policy_name}...")

    episode_rewards = []
    action_counts = [0, 0]  # [Stay, Switch]
    decision_counts = []

    for episode in range(num_episodes):
        # 채널 상태 초기화
        for ch in channels:
            ch.intra_occupied = False
            ch.intra_end_slot = 0
            ch.obss_traffic = []
            ch.occupied_remain = 0
            ch.obss_remain = 0

        # STA 생성 (training과 동일)
        stas = []
        for config in stas_config:
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=channels[config["channel_id"]],
                npca_channel=channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=config.get("npca_enabled", False),
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", 33),
                random_ppdu=random_ppdu,
                learner=None,
                num_slots_per_episode=num_slots_per_episode
            )

            # NPCA 지원 STA에 고정 정책 적용
            if config.get("npca_enabled", False):
                sta._fixed_action = policy_func
                sta.decision_count = 0  # 결정 횟수 추적
                sta.action_history = []  # 액션 기록 추적 (Random 정책용)

            stas.append(sta)

        # 시뮬레이션 실행
        simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
        simulator.memory = None  # baseline 평가에서는 memory 불필요
        simulator.device = torch.device("cpu")
        simulator.run()

        # 에피소드별 총 보상 수집 (training과 동일한 방식)
        total_reward = 0
        total_decisions = 0
        episode_actions = [0, 0]

        for sta in stas:
            if sta.npca_enabled:
                total_reward += sta.new_episode_reward
                decisions = getattr(sta, 'decision_count', 0)
                total_decisions += decisions

                # 정책에 따른 액션 분포
                if policy_name == "Primary-Only":
                    episode_actions[0] += decisions
                elif policy_name == "NPCA-Only":
                    episode_actions[1] += decisions
                elif policy_name == "Random":
                    # Random 정책의 경우 실제 액션 기록을 추적해야 함
                    action_history = getattr(sta, 'action_history', [])
                    for action in action_history:
                        episode_actions[action] += 1

            sta.new_episode_reward = 0.0

        episode_rewards.append(total_reward)
        action_counts[0] += episode_actions[0]
        action_counts[1] += episode_actions[1]
        decision_counts.append(total_decisions)

    # 통계 계산
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_decisions = np.mean(decision_counts)

    total_actions = sum(action_counts)
    action_probs = [count/max(total_actions, 1) for count in action_counts]

    return {
        'policy_name': policy_name,
        'episode_rewards': episode_rewards,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'action_distribution': action_probs,
        'avg_decisions': avg_decisions,
        'total_decisions': total_actions
    }

def run_baseline_comparison(obss_duration=100, ppdu_variant='medium',
                          num_episodes=100, num_slots_per_episode=11111, random_ppdu=False):
    """모든 baseline 정책 비교"""

    ppdu_description = "Random (20-200 slots)" if random_ppdu else f"{ppdu_variant} (fixed)"

    print(f"\n{'='*60}")
    print(f"BASELINE POLICY COMPARISON")
    print(f"OBSS Duration: {obss_duration}, PPDU: {ppdu_description}")
    print(f"Episodes: {num_episodes}, Slots per episode: {num_slots_per_episode}")
    print(f"STA Configuration: CH0={DEFAULT_NUM_STAS_CH0}, CH1={DEFAULT_NUM_STAS_CH1}")
    print(f"{'='*60}")

    # 환경 설정 (training과 동일)
    channels, stas_config = create_baseline_config(obss_duration, ppdu_variant, random_ppdu)

    results = []

    # Primary-Only 정책 평가
    primary_result = evaluate_baseline_policy(
        FixedPolicy.primary_only, "Primary-Only",
        channels, stas_config, num_episodes, num_slots_per_episode, random_ppdu
    )
    results.append(primary_result)

    # NPCA-Only 정책 평가
    npca_result = evaluate_baseline_policy(
        FixedPolicy.npca_only, "NPCA-Only",
        channels, stas_config, num_episodes, num_slots_per_episode, random_ppdu
    )
    results.append(npca_result)

    # Random 정책 평가
    random_result = evaluate_baseline_policy(
        FixedPolicy.random, "Random",
        channels, stas_config, num_episodes, num_slots_per_episode, random_ppdu
    )
    results.append(random_result)

    # 결과 출력
    print(f"\nBaseline Results:")
    print(f"{'Policy':<15} {'Avg Reward':<12} {'Std':<8} {'Stay%':<8} {'Switch%':<8}")
    print("-" * 55)

    for result in results:
        stay_pct = result['action_distribution'][0] * 100
        switch_pct = result['action_distribution'][1] * 100
        print(f"{result['policy_name']:<15} {result['avg_reward']:<12.2f} "
              f"{result['std_reward']:<8.2f} {stay_pct:<8.0f} {switch_pct:<8.0f}")

    return results

def evaluate_drl_policy(model_path, channels, stas_config, num_episodes=100, num_slots_per_episode=11111, random_ppdu=False):
    """DRL 정책을 baseline과 동일한 환경에서 평가"""

    try:
        # 모델 로드
        device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        from drl_framework.network import DQN
        from drl_framework.train import SemiMDPLearner

        # DQN 모델 생성 및 로드
        policy_net = DQN(n_observations=4, n_actions=2).to(device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        policy_net.eval()

        # Mock learner 생성 (평가용)
        class MockLearner:
            def __init__(self, policy_net, device):
                self.policy_net = policy_net
                self.device = device
                self.memory = None
                self.steps_done = 0

            def select_action(self, state_tensor):
                with torch.no_grad():
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    return q_values.max(1)[1].item()

        mock_learner = MockLearner(policy_net, device)

        print(f"Evaluating DRL Model...")

        episode_rewards = []
        action_counts = [0, 0]  # [Stay, Switch]
        decision_counts = []

        for episode in range(num_episodes):
            # 채널 상태 초기화
            for ch in channels:
                ch.intra_occupied = False
                ch.intra_end_slot = 0
                ch.obss_traffic = []
                ch.occupied_remain = 0
                ch.obss_remain = 0

            # STA 생성 (baseline과 동일)
            stas = []
            for config in stas_config:
                sta = STA(
                    sta_id=config["sta_id"],
                    channel_id=config["channel_id"],
                    primary_channel=channels[config["channel_id"]],
                    npca_channel=channels[0] if config["channel_id"] == 1 else None,
                    npca_enabled=config.get("npca_enabled", False),
                    radio_transition_time=config.get("radio_transition_time", 1),
                    ppdu_duration=config.get("ppdu_duration", 33),
                    random_ppdu=random_ppdu,
                    learner=mock_learner if config.get("npca_enabled", False) else None,
                    num_slots_per_episode=num_slots_per_episode
                )

                # 결정 추적을 위한 설정
                if config.get("npca_enabled", False):
                    sta.decision_count = 0
                    sta.action_history = []  # 액션 기록용

                stas.append(sta)

            # 시뮬레이션 실행
            simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
            simulator.memory = mock_learner.memory
            simulator.device = device
            simulator.run()

            # 에피소드별 총 보상 수집
            total_reward = 0
            total_decisions = 0
            episode_actions = [0, 0]

            for sta in stas:
                if sta.npca_enabled:
                    total_reward += sta.new_episode_reward
                    decisions = getattr(sta, 'decision_count', 0)
                    total_decisions += decisions

                    # 액션 기록에서 실제 분포 계산 (추정이 아닌 실제값)
                    action_history = getattr(sta, 'action_history', [])
                    for action in action_history:
                        episode_actions[action] += 1

                sta.new_episode_reward = 0.0

            episode_rewards.append(total_reward)
            action_counts[0] += episode_actions[0]
            action_counts[1] += episode_actions[1]
            decision_counts.append(total_decisions)

        # 통계 계산
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_decisions = np.mean(decision_counts)

        total_actions = sum(action_counts)
        action_probs = [count/max(total_actions, 1) for count in action_counts]

        return {
            'policy_name': 'DRL',
            'episode_rewards': episode_rewards,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'action_distribution': action_probs,
            'avg_decisions': avg_decisions,
            'total_decisions': total_actions
        }

    except Exception as e:
        print(f"Error evaluating DRL model: {e}")
        return None

def compare_with_drl_model(obss_duration=100, ppdu_variant='medium', model_path=None, random_ppdu=True):
    """DRL 모델과 baseline 비교 (동일한 환경에서 실제 평가)"""

    if model_path is None:
        # 가능한 모델 경로들
        model_paths = [
            f"./density_comparison_results/ch0_{DEFAULT_NUM_STAS_CH0}_ch1_{DEFAULT_NUM_STAS_CH1}/model.pth",
            f"./density_comparison_results/obss_{obss_duration}_slots/model.pth"
        ]

        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

    # 환경 설정 (baseline과 동일)
    channels, stas_config = create_baseline_config(obss_duration, ppdu_variant, random_ppdu)

    # Baseline 결과
    baseline_results = run_baseline_comparison(obss_duration, ppdu_variant,
                                             num_episodes=100, num_slots_per_episode=11111,
                                             random_ppdu=random_ppdu)

    # DRL 모델 결과 (동일한 환경에서 실제 평가)
    drl_result = None
    if model_path and os.path.exists(model_path):
        print(f"\n🤖 Evaluating DRL model: {model_path}")
        drl_result = evaluate_drl_policy(
            model_path, channels, stas_config,
            num_episodes=100, num_slots_per_episode=11111, random_ppdu=True
        )

        if drl_result:
            print(f"DRL: Avg Reward = {drl_result['avg_reward']:.2f} ± {drl_result['std_reward']:.2f}")
            print(f"  Action Dist: Stay={drl_result['action_distribution'][0]:.2f}, Switch={drl_result['action_distribution'][1]:.2f}")
    else:
        print("⚠️ No DRL model found for evaluation")

    # 비교 시각화
    all_results = baseline_results.copy()
    if drl_result:
        all_results.append(drl_result)

    create_comparison_plot(all_results, obss_duration)

    return baseline_results, drl_result

def create_comparison_plot(all_results, obss_duration=100):
    """모든 정책 결과 비교 플롯"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 평균 보상 비교
    ax1 = axes[0]
    policies = [r['policy_name'] for r in all_results]
    rewards = [r['avg_reward'] for r in all_results]
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'gold'][:len(all_results)]

    bars = ax1.bar(policies, rewards, color=colors)
    ax1.set_title(f'Policy Comparison (OBSS Duration: {obss_duration})')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)

    # 막대 위에 값 표시
    for bar, reward in zip(bars, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rewards)*0.01,
                f'{reward:.2f}', ha='center', fontweight='bold')

    # 액션 분포 비교 (모든 정책)
    ax2 = axes[1]
    policy_names = [r['policy_name'] for r in all_results]
    stay_probs = [r['action_distribution'][0] for r in all_results]
    switch_probs = [r['action_distribution'][1] for r in all_results]

    x_pos = np.arange(len(policy_names))
    width = 0.35

    ax2.bar(x_pos - width/2, stay_probs, width, label='Stay Primary', color='lightblue')
    ax2.bar(x_pos + width/2, switch_probs, width, label='Switch NPCA', color='lightsalmon')

    ax2.set_title('Action Distribution (All Policies)')
    ax2.set_ylabel('Action Probability')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(policy_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    os.makedirs("./baseline_comparison_results", exist_ok=True)

    # ax1 plot을 별도로 PNG와 EPS로 저장
    fig1, ax1_single = plt.subplots(figsize=(8, 6))
    bars = ax1_single.bar(policies, rewards, color=colors)
    # ax1_single.set_title(f'Policy Comparison (OBSS Duration: {obss_duration})')
    ax1_single.set_ylabel('Average Reward')
    ax1_single.grid(True, alpha=0.3)

    # 막대 위에 값 표시
    for bar, reward in zip(bars, rewards):
        ax1_single.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rewards)*0.01,
                       f'{reward:.2f}', ha='center', fontweight='bold')

    # ax1 plot PNG 저장
    fig1.savefig(f"./baseline_comparison_results/policy_comparison_obss_{obss_duration}.png",
                 dpi=300, bbox_inches='tight')

    # ax1 plot EPS 저장
    fig1.savefig(f"./baseline_comparison_results/policy_comparison_obss_{obss_duration}.eps",
                 format='eps', bbox_inches='tight')

    plt.close(fig1)

    # 전체 플롯 저장
    plt.savefig(f"./baseline_comparison_results/baseline_comparison_obss_{obss_duration}.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nComparison plot saved to ./baseline_comparison_results/")
    print(f"ax1 plot saved as PNG and EPS: policy_comparison_obss_{obss_duration}.*")

def main():
    """메인 실행 함수"""
    import sys

    # 커맨드라인 인자 처리
    obss_duration = 100
    if len(sys.argv) > 1:
        try:
            obss_duration = int(sys.argv[1])
        except ValueError:
            print("Invalid OBSS duration. Using default (100).")

    print(f"🔍 Starting baseline evaluation...")
    print(f"Testing environment: OBSS Duration = {obss_duration} slots")

    # Baseline과 DRL 모델 비교 (랜덤 PPDU 사용)
    baseline_results, drl_result = compare_with_drl_model(obss_duration, random_ppdu=True)

    # 요약 출력
    print(f"\n{'🎯 SUMMARY:'}")
    print("-" * 50)

    best_baseline = max(baseline_results, key=lambda x: x['avg_reward'])
    print(f"Best Baseline: {best_baseline['policy_name']} ({best_baseline['avg_reward']:.2f})")

    if drl_result:
        print(f"DRL Model: {drl_result['avg_reward']:.2f}")
        improvement = drl_result['avg_reward'] - best_baseline['avg_reward']
        if improvement > 0:
            print(f"🏆 DRL improves over best baseline by {improvement:.2f} points")
        else:
            print(f"⚠️ DRL underperforms best baseline by {-improvement:.2f} points")
    else:
        print("⚠️ No DRL model found for comparison")

    print(f"\n✅ Baseline evaluation complete!")

if __name__ == "__main__":
    main()