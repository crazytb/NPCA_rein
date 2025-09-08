#!/usr/bin/env python3
"""
OBSS 발생 상황에서 디버깅
"""

import torch
from drl_framework.random_access import Channel, STA, Simulator

def force_obss_test():
    """OBSS를 강제로 발생시켜 테스트"""
    print("OBSS 강제 발생 테스트...")
    
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))
    ]
    
    # 강제로 OBSS 생성 (slot 10부터 100 슬롯 동안)
    channels[1].obss_traffic.append(("force_obss", 10, 100, 999))
    
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
    
    # Always NPCA 전략
    sta._fixed_action = 1
    decision_log = []
    sta.decision_log = decision_log
    sta.current_episode = 0
    
    print(f"강제 OBSS: slot 10-110")
    print(f"초기 상태: {sta.state.name}")
    
    decision_count = 0
    
    for slot in range(200):
        # 채널 업데이트
        for ch in channels:
            ch.update(slot)
        
        # 슬롯 10-15 구간 상세 로그
        if 10 <= slot <= 15:
            print(f"Slot {slot}: State={sta.state.name}, Primary busy={channels[1].is_busy(slot)}, Opt_active={getattr(sta, '_opt_active', False)}")
            
            # 결정 조건 확인
            if (sta.state.name == "PRIMARY_BACKOFF" and 
                channels[1].is_busy(slot) and 
                not getattr(sta, '_opt_active', False)):
                decision_count += 1
                print(f"*** 결정 발생! Count: {decision_count}")
        
        # STA 스텝
        sta.step(slot)
        sta.state = sta.next_state
    
    print(f"총 결정 횟수: {decision_count}")
    print(f"로그된 결정 수: {len(decision_log)}")
    print(f"최종 보상: {sta.episode_reward}")
    
    if decision_log:
        print("결정 로그:")
        for i, log in enumerate(decision_log):
            print(f"  {i+1}: Slot={log['slot']}, Action={log['action']}, OBSS_remain={log['primary_channel_obss_occupied_remained']}")

def test_direct_comparison_issue():
    """direct_comparison.py의 문제 확인"""
    print("\n" + "="*50)
    print("Direct comparison 문제 확인")
    print("="*50)
    
    # 실제 비교 함수와 동일한 방식으로 테스트
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))
    ]
    
    # 10개 STA 생성 (direct_comparison.py와 동일)
    stas_config = []
    for i in range(10):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    episode_rewards = []
    
    for episode in range(3):  # 3 에피소드만 테스트
        print(f"\nEpisode {episode}:")
        
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
                learner=None
            )
            sta._fixed_action = 1  # Always NPCA
            stas.append(sta)
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=stas)
        simulator.run()
        
        # 보상 수집
        total_reward = sum(sta.episode_reward for sta in stas)
        normalized_reward = total_reward / 100  # direct_comparison.py와 동일한 정규화
        episode_rewards.append(normalized_reward)
        
        print(f"  Raw total reward: {total_reward}")
        print(f"  Normalized reward: {normalized_reward}")
        print(f"  Individual rewards: {[sta.episode_reward for sta in stas[:3]]}...")  # 처음 3개만 출력
    
    print(f"\n최종 에피소드 보상들: {episode_rewards}")
    print(f"평균: {sum(episode_rewards)/len(episode_rewards):.3f}")

if __name__ == "__main__":
    force_obss_test()
    test_direct_comparison_issue()