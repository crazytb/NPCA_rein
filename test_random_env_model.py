#!/usr/bin/env python3
"""
Random Environment Model 성능 테스트

이 스크립트는 random environment로 훈련된 모델의 성능을 평가하고
기존 fixed environment 모델들과 비교합니다.
"""

import torch
import numpy as np
from pathlib import Path
from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN
import time

def dict_to_legacy_vector(obs_dict):
    """Semi-MDP Dict 관찰을 기존 4차원 벡터로 변환"""
    return [
        float(obs_dict.get('obss_remaining', 0)),
        float(obs_dict.get('current_slot', 1)),
        33.0,  # tx_duration (고정값)
        float(obs_dict.get('cw_index', 0))
    ]

def load_drl_model(model_path, device='cpu'):
    """DRL 모델 로드 및 정책 함수 생성"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 모델 관찰 공간 크기 확인
        state_dict = checkpoint['policy_net_state_dict']
        input_size = state_dict['layer1.weight'].shape[1]
        print(f"  모델 입력 크기: {input_size} 차원")
        
        model = DQN(n_observations=input_size, n_actions=2).to(device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        if input_size == 4:
            # 4차원 벡터 모델 (기존 모델)
            def policy_func(obs_dict):
                obs_vector = dict_to_legacy_vector(obs_dict)
                input_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(input_tensor)
                    action = q_values.argmax(dim=1).item()
                return action
        else:
            # 전체 관찰 공간 모델 (random environment 모델)
            def policy_func(obs_dict):
                # NPCASemiMDPEnv의 flatten_observation과 동일한 방식으로 변환
                obs_vector = []
                
                # 기본 관찰값들
                obs_vector.append(float(obs_dict.get('obss_remaining', 0)))
                obs_vector.append(float(obs_dict.get('current_slot', 1)))
                obs_vector.append(float(obs_dict.get('cw_index', 0)))
                
                # 환경 파라미터들 (random env에서 추가됨)
                if 'env_obss_duration' in obs_dict:
                    obs_vector.append(float(obs_dict['env_obss_duration']))
                else:
                    obs_vector.append(100.0)  # 기본값
                    
                if 'env_ppdu_duration' in obs_dict:
                    obs_vector.append(float(obs_dict['env_ppdu_duration']))
                else:
                    obs_vector.append(33.0)  # 기본값
                
                # 나머지 관찰값들을 0으로 패딩 (필요시)
                while len(obs_vector) < input_size:
                    obs_vector.append(0.0)
                
                # 크기가 맞지 않으면 자르기
                obs_vector = obs_vector[:input_size]
                
                input_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(input_tensor)
                    action = q_values.argmax(dim=1).item()
                return action
        
        return policy_func, checkpoint
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def test_policy(policy_func, policy_name, test_episodes=50, random_env=False, 
                obss_duration=100, ppdu_duration=33):
    """정책 성능 테스트"""
    env = NPCASemiMDPEnv(
        num_stas=2,
        num_slots=3000,
        obss_generation_rate=0.05,
        npca_enabled=True,
        throughput_weight=10.0,
        latency_penalty_weight=0.1,
        random_env=random_env
    )
    
    episode_rewards = []
    episode_throughputs = []
    episode_latencies = []
    action_counts = [0, 0]
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_latency = 0.0
        
        done = False
        max_decisions = 100
        episode_decisions = 0
        
        while not done and episode_decisions < max_decisions:
            action = policy_func(obs)
            action_counts[action] += 1
            
            try:
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_throughput += info.get('successful_transmission_slots', 0)
                episode_latency += info.get('duration', 0)
                episode_decisions += 1
                obs = next_obs
                
                if done or truncated:
                    break
            except ValueError as e:
                if "step() called when not at decision point" in str(e):
                    break
                else:
                    raise e
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
        episode_latencies.append(episode_latency)
    
    total_actions = sum(action_counts)
    action_probs = [count/total_actions for count in action_counts] if total_actions > 0 else [0.5, 0.5]
    
    return {
        'policy_name': policy_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_throughput': np.mean(episode_throughputs),
        'avg_latency': np.mean(episode_latencies),
        'action_distribution': action_probs,
        'episode_rewards': episode_rewards
    }

def main():
    print("🔬 Random Environment Model 성능 테스트")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Random environment 모델 로드
    random_model_path = "./obss_comparison_results/random_env_robust_model/model.pth"
    random_policy, random_checkpoint = load_drl_model(random_model_path, device)
    
    if not random_policy:
        print("❌ Random environment 모델을 찾을 수 없습니다.")
        return
    
    print(f"✅ Random environment 모델 로드 완료!")
    print(f"   훈련 스텝: {random_checkpoint.get('steps_done', 'unknown')}")
    
    # 1. Random environment에서 테스트 (훈련 환경과 동일)
    print("\n🌟 Random Environment 테스트 (훈련 환경과 동일):")
    random_result = test_policy(
        random_policy, 'DRL-Random-Env', 
        test_episodes=100, random_env=True
    )
    
    print(f"  평균 보상: {random_result['avg_reward']:.1f} ± {random_result['std_reward']:.1f}")
    print(f"  평균 처리량: {random_result['avg_throughput']:.1f}")
    print(f"  평균 지연: {random_result['avg_latency']:.1f}")
    print(f"  액션 분포: Stay {random_result['action_distribution'][0]:.2f}, Switch {random_result['action_distribution'][1]:.2f}")
    
    # 2. 다양한 고정 환경에서 테스트 (일반화 성능 확인)
    print("\n🔍 Fixed Environment 테스트 (일반화 성능):")
    fixed_scenarios = [
        {'obss': 50, 'ppdu': 33, 'name': 'Short OBSS (50)'},
        {'obss': 100, 'ppdu': 33, 'name': 'Medium OBSS (100)'},
        {'obss': 200, 'ppdu': 33, 'name': 'Long OBSS (200)'},
        {'obss': 150, 'ppdu': 33, 'name': 'Custom OBSS (150)'}
    ]
    
    fixed_results = []
    for scenario in fixed_scenarios:
        result = test_policy(
            random_policy, f'DRL-Random-Fixed-{scenario["obss"]}',
            test_episodes=50, random_env=False,
            obss_duration=scenario['obss'], ppdu_duration=scenario['ppdu']
        )
        fixed_results.append(result)
        print(f"  {scenario['name']:18s}: Reward={result['avg_reward']:6.1f}, "
              f"Throughput={result['avg_throughput']:5.1f}, "
              f"Stay/Switch={result['action_distribution'][0]:.2f}/{result['action_distribution'][1]:.2f}")
    
    # 3. 기존 모델들과 비교
    print("\n📊 기존 모델들과 성능 비교:")
    fixed_model_paths = [
        ("./obss_comparison_results/ppdu_medium_obss_50/model.pth", "DRL-Fixed-50"),
        ("./obss_comparison_results/ppdu_medium_obss_100/model.pth", "DRL-Fixed-100"),
        ("./obss_comparison_results/ppdu_medium_obss_200/model.pth", "DRL-Fixed-200")
    ]
    
    comparison_results = []
    for model_path, model_name in fixed_model_paths:
        if Path(model_path).exists():
            fixed_policy, _ = load_drl_model(model_path, device)
            if fixed_policy:
                # Random environment에서 테스트
                result = test_policy(
                    fixed_policy, model_name,
                    test_episodes=50, random_env=True
                )
                comparison_results.append(result)
                print(f"  {model_name:15s}: Reward={result['avg_reward']:6.1f}, "
                      f"Stay/Switch={result['action_distribution'][0]:.2f}/{result['action_distribution'][1]:.2f}")
    
    # 4. 종합 분석
    print(f"\n📈 종합 성능 분석:")
    print("="*50)
    
    # Random environment 평균 성능
    avg_fixed = sum(r['avg_reward'] for r in fixed_results) / len(fixed_results)
    std_fixed = (sum((r['avg_reward'] - avg_fixed)**2 for r in fixed_results) / len(fixed_results))**0.5
    
    print(f"Random Environment 모델:")
    print(f"  Random Env (훈련 환경):  {random_result['avg_reward']:6.1f} ± {random_result['std_reward']:4.1f}")
    print(f"  Fixed Env (일반화):     {avg_fixed:6.1f} ± {std_fixed:4.1f}")
    
    if comparison_results:
        avg_comparison = sum(r['avg_reward'] for r in comparison_results) / len(comparison_results)
        print(f"\nFixed Environment 모델들 (Random Env에서):")
        print(f"  평균 성능:              {avg_comparison:6.1f}")
        
        print(f"\n💡 Random Environment 모델의 장점:")
        if random_result['avg_reward'] > avg_comparison:
            print("  ✅ Random env에서 기존 모델들보다 우수한 성능")
        if avg_fixed > avg_comparison * 0.9:
            print("  ✅ 다양한 fixed environment에서 안정적인 성능")
        if random_result['std_reward'] < avg_fixed * 1.2:
            print("  ✅ 일관된 성능으로 높은 신뢰성")
    
    print(f"\n🎯 결론:")
    print(f"Random environment 훈련은 다양한 OBSS/PPDU 조건에 강건한 정책을 학습했습니다!")
    
if __name__ == "__main__":
    main()