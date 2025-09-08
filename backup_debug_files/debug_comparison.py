#!/usr/bin/env python3
"""
비교 테스트 디버깅 스크립트
"""

import torch
from drl_framework.random_access import Channel, STA, Simulator

def debug_single_episode():
    """단일 에피소드 디버깅"""
    print("단일 에피소드 디버깅 시작...")
    
    # 설정 생성
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))
    ]
    
    # STA 1개만 테스트
    sta = STA(
        sta_id=0,
        channel_id=1,
        primary_channel=channels[1],
        npca_channel=channels[0],
        npca_enabled=True,
        ppdu_duration=33,
        radio_transition_time=1,
        learner=None
    )
    
    # Always NPCA 전략 설정
    sta._fixed_action = 1
    
    # 시뮬레이터 생성
    simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
    
    print(f"초기 상태: {sta.state.name}")
    print(f"Primary channel (1) OBSS rate: {channels[1].obss_generation_rate}")
    print(f"NPCA channel (0) OBSS rate: {channels[0].obss_generation_rate}")
    
    decision_count = 0
    reward_events = []
    
    # 수동으로 슬롯별 시뮬레이션
    for slot in range(200):
        # 채널 업데이트
        for ch in channels:
            ch.update(slot)
        
        # STA 상태 확인
        if slot % 50 == 0:
            print(f"Slot {slot}: State={sta.state.name}, Episode_reward={sta.episode_reward:.3f}")
            print(f"  Primary ch busy: {channels[1].is_busy(slot)}")
            print(f"  NPCA ch busy: {channels[0].is_busy(slot)}")
        
        # 결정 시점 확인
        if (sta.state.name == "PRIMARY_BACKOFF" and 
            channels[1].is_busy(slot) and 
            not getattr(sta, '_opt_active', False)):
            decision_count += 1
            print(f"*** 결정 시점 {decision_count} at slot {slot}")
            print(f"    OBSS remaining: {channels[1].get_obss_remaining(slot)}")
            print(f"    Action: {sta._fixed_action} (Always NPCA)")
        
        # 보상 발생 확인
        prev_reward = sta.episode_reward
        
        # STA 스텝
        sta.step(slot)
        
        if sta.episode_reward > prev_reward:
            reward_delta = sta.episode_reward - prev_reward
            reward_events.append((slot, reward_delta, sta.state.name))
            print(f"*** 보상 발생 at slot {slot}: +{reward_delta:.3f}, State={sta.state.name}")
        
        # 상태 전이
        sta.state = sta.next_state
    
    print(f"\n=== 에피소드 완료 ===")
    print(f"총 결정 횟수: {decision_count}")
    print(f"총 보상 이벤트: {len(reward_events)}")
    print(f"최종 에피소드 보상: {sta.episode_reward:.3f}")
    print(f"보상 이벤트 상세: {reward_events}")

def debug_drl_policy():
    """DRL 정책 디버깅"""
    print("\nDRL 정책 디버깅 시작...")
    
    # 모델 로드
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    from drl_framework.train import SemiMDPLearner
    learner = SemiMDPLearner(n_observations=4, n_actions=2, device='cpu')
    learner.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    learner.steps_done = checkpoint['steps_done']
    learner.policy_net.eval()
    
    # 설정 생성
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))
    ]
    
    # STA 1개만 테스트
    sta = STA(
        sta_id=0,
        channel_id=1,
        primary_channel=channels[1],
        npca_channel=channels[0],
        npca_enabled=True,
        ppdu_duration=33,
        radio_transition_time=1,
        learner=learner
    )
    
    # 시뮬레이터 생성
    simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
    simulator.memory = learner.memory
    simulator.device = 'cpu'
    simulator.run()
    
    print(f"DRL 정책 최종 보상: {sta.episode_reward:.3f}")

if __name__ == "__main__":
    debug_single_episode()
    debug_drl_policy()