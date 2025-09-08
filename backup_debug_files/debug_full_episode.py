#!/usr/bin/env python3
"""
Always Primary의 전체 에피소드 흐름 디버깅
"""

from drl_framework.random_access import Channel, STA, Simulator

def debug_full_episode():
    """Always Primary의 전체 에피소드 흐름 추적"""
    print("Always Primary 전체 에피소드 디버깅...")
    
    # 현실적인 OBSS 설정 (simple_fixed_test.py와 동일)
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),
        Channel(channel_id=1, obss_generation_rate=0.6, obss_duration_range=(80, 150))
    ]
    
    # STA 생성
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
    
    # Always Primary 전략
    sta._fixed_action = 0
    decision_log = []
    sta.decision_log = decision_log
    sta.current_episode = 0
    
    print(f"설정:")
    print(f"  Fixed action: {sta._fixed_action} (StayPrimary)")
    print(f"  PPDU duration: {sta.ppdu_duration}")
    print(f"  OBSS rate: {channels[1].obss_generation_rate}")
    print(f"  OBSS duration: {channels[1].obss_duration_range}")
    
    # Simulator.run() 사용
    simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
    simulator.run()
    
    print(f"\n=== 최종 결과 ===")
    print(f"점유 시간: {sta.channel_occupancy_time}")
    print(f"Episode reward: {sta.episode_reward}")
    print(f"결정 횟수: {len(decision_log)}")
    
    if decision_log:
        print("\n=== 결정 로그 ===")
        for i, log in enumerate(decision_log):
            print(f"결정 {i+1}: Slot={log['slot']}, Action={log['action']}, OBSS_remain={log['primary_channel_obss_occupied_remained']}")
    
    # 전송 성공 여부 분석
    print(f"\n=== 전송 성공 분석 ===")
    print("Always Primary의 문제점:")
    print("1. OBSS 발생 시 StayPrimary 선택")
    print("2. PRIMARY_FROZEN 상태로 이동 (OBSS 대기)")
    print("3. OBSS 종료 후 PRIMARY_BACKOFF로 복귀")
    print("4. 하지만 높은 OBSS 발생률로 인해 전송 기회 부족")
    print("5. 200슬롯 에피소드에서 OBSS 길이(80-150)가 너무 길어서 전송 불가능")

def compare_with_lower_obss():
    """더 낮은 OBSS 발생률에서 Always Primary 성능 확인"""
    print("\n" + "="*60)
    print("낮은 OBSS 발생률에서 Always Primary 테스트")
    print("="*60)
    
    # 낮은 OBSS 발생률
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),
        Channel(channel_id=1, obss_generation_rate=0.1, obss_duration_range=(20, 40))  # 낮은 발생률, 짧은 지속시간
    ]
    
    results = {}
    strategies = [
        ("Always Primary", 0),
        ("Always NPCA", 1)
    ]
    
    for strategy_name, action in strategies:
        episode_rewards = []
        
        for episode in range(5):
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
            
            sta._fixed_action = action
            decision_log = []
            sta.decision_log = decision_log
            sta.current_episode = episode
            
            simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
            simulator.run()
            
            episode_rewards.append(sta.episode_reward)
        
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        results[strategy_name] = avg_reward
        print(f"{strategy_name}: 평균 {avg_reward:.1f}% 점유율")
    
    return results

if __name__ == "__main__":
    debug_full_episode()
    compare_with_lower_obss()