#!/usr/bin/env python3
"""
NPCA 액션 선택 요인 분석을 위한 스크립트
논문 작성을 위한 주요 의사결정 요인 식별
"""

import torch
import numpy as np
from drl_framework.train import train_semi_mdp
from drl_framework.random_access import Channel
from main_semi_mdp_training import create_training_config

def analyze_decision_factors():
    """각 상태 특성이 액션 선택에 미치는 영향을 분석"""
    
    print("="*60)
    print("NPCA 액션 선택 요인 분석")
    print("="*60)
    
    # 학습된 모델 로드 시도
    try:
        checkpoint = torch.load("./semi_mdp_results/semi_mdp_model.pth", map_location='cpu')
        print("✓ 기존 학습된 모델을 로드했습니다.")
    except:
        print("✗ 학습된 모델이 없습니다. 먼저 학습을 진행해주세요.")
        return
    
    # 테스트 시나리오 생성
    test_scenarios = [
        # 시나리오 1: OBSS 점유 시간이 다른 경우
        {"primary_channel_obss_occupied_remained": 5, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "짧은 OBSS (5 slots)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "중간 OBSS (20 slots)"},
        {"primary_channel_obss_occupied_remained": 40, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "긴 OBSS (40 slots)"},
        
        # 시나리오 2: Radio transition time이 다른 경우
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "빠른 전환 (1 slot)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 5, "tx_duration": 33, "cw_index": 0, "scenario": "느린 전환 (5 slots)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 10, "tx_duration": 33, "cw_index": 0, "scenario": "매우 느린 전환 (10 slots)"},
        
        # 시나리오 3: Contention Window가 다른 경우
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "낮은 경쟁 (CW=0)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 3, "scenario": "중간 경쟁 (CW=3)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 6, "scenario": "높은 경쟁 (CW=6)"},
        
        # 시나리오 4: TX duration이 다른 경우
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 10, "cw_index": 0, "scenario": "짧은 전송 (10 slots)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 33, "cw_index": 0, "scenario": "일반 전송 (33 slots)"},
        {"primary_channel_obss_occupied_remained": 20, "radio_transition_time": 1, "tx_duration": 60, "cw_index": 0, "scenario": "긴 전송 (60 slots)"},
    ]
    
    # DQN 네트워크 초기화
    from drl_framework.network import DQN
    from drl_framework.random_access import STA
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(4, 2).to(device)  # 4 features, 2 actions
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()
    
    print(f"\n📊 테스트 시나리오별 액션 선택 분석:")
    print(f"{'시나리오':<25} {'Action 0 (Stay)':<15} {'Action 1 (NPCA)':<15} {'선택된 액션':<10}")
    print("-" * 70)
    
    # 각 시나리오에 대해 액션 예측
    for scenario in test_scenarios:
        # 정규화된 상태 벡터 생성
        obs_vec = [
            scenario["primary_channel_obss_occupied_remained"] / 1024.0,
            scenario["radio_transition_time"] / 1024.0, 
            scenario["tx_duration"] / 1024.0,
            scenario["cw_index"] / 8.0
        ]
        
        state_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)
            selected_action = q_values.max(1)[1].item()
            
        prob_stay = action_probs[0][0].item()
        prob_npca = action_probs[0][1].item()
        action_text = "Stay" if selected_action == 0 else "NPCA"
        
        print(f"{scenario['scenario']:<25} {prob_stay:<15.3f} {prob_npca:<15.3f} {action_text:<10}")
    
    # 각 특성별 영향 분석
    print(f"\n📈 특성별 영향도 분석:")
    analyze_feature_sensitivity(policy_net, device)

def analyze_feature_sensitivity(policy_net, device):
    """각 특성의 변화에 대한 액션 확률의 민감도 분석"""
    
    # 기준 상태
    baseline = [20/1024.0, 1/1024.0, 33/1024.0, 0/8.0]  # 정규화된 값
    
    features = [
        "OBSS 남은 시간",
        "Radio 전환 시간", 
        "전송 지속 시간",
        "Contention Window"
    ]
    
    for i, feature_name in enumerate(features):
        print(f"\n🔍 {feature_name} 변화에 따른 액션 확률:")
        
        if i == 0:  # OBSS 남은 시간
            test_values = [1, 5, 10, 20, 30, 40]
            normalizer = 1024.0
        elif i == 1:  # Radio 전환 시간
            test_values = [1, 2, 5, 10, 15]
            normalizer = 1024.0
        elif i == 2:  # 전송 지속 시간
            test_values = [10, 20, 33, 50, 100]
            normalizer = 1024.0
        else:  # CW index
            test_values = [0, 1, 2, 3, 4, 5, 6]
            normalizer = 8.0
            
        for value in test_values:
            test_state = baseline.copy()
            test_state[i] = value / normalizer
            
            state_tensor = torch.tensor(test_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action_probs = torch.softmax(q_values, dim=1)
                selected_action = q_values.max(1)[1].item()
                
            prob_stay = action_probs[0][0].item()
            prob_npca = action_probs[0][1].item()
            action_text = "Stay" if selected_action == 0 else "NPCA"
            
            print(f"  {value:>3} → Stay: {prob_stay:.3f}, NPCA: {prob_npca:.3f} [{action_text}]")

if __name__ == "__main__":
    analyze_decision_factors()