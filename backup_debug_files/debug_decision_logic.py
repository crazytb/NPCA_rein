#!/usr/bin/env python3
"""
고정 전략에서 결정 로직 실행 여부 디버깅
"""

from drl_framework.random_access import Channel, STA, Simulator

def debug_decision_logic():
    """고정 전략의 결정 로직 실행 여부 확인"""
    print("고정 전략 결정 로직 디버깅...")
    
    # 높은 OBSS 발생률로 결정 시점 보장
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  
        Channel(channel_id=1, obss_generation_rate=1.0, obss_duration_range=(10, 20))  # 짧은 OBSS로 테스트
    ]
    
    # 강제로 OBSS 생성
    channels[1].obss_traffic.append(("test_obss", 5, 50, 999))
    
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
    print(f"  Fixed action: {sta._fixed_action}")
    print(f"  NPCA enabled: {sta.npca_enabled}")
    print(f"  Has learner: {sta.learner is not None}")
    print(f"  Has _fixed_action: {hasattr(sta, '_fixed_action')}")
    
    print(f"\n=== 수동 시뮬레이션 (결정 로직 추적) ===")
    option_starts = []
    
    for slot in range(100):
        # 채널 업데이트
        for ch in channels:
            ch.update(slot)
        
        primary_busy = channels[1].is_busy(slot)
        obss_busy = channels[1].is_busy_by_obss(slot)
        
        # 결정 조건 체크
        if (sta.state.name == "PRIMARY_BACKOFF" and 
            primary_busy and obss_busy and 
            not getattr(sta, '_opt_active', False)):
            
            print(f"*** 결정 조건 만족 at slot {slot}")
            print(f"    State: {sta.state.name}")
            print(f"    Primary busy: {primary_busy}")
            print(f"    OBSS busy: {obss_busy}")
            print(f"    NPCA enabled: {sta.npca_enabled}")
            print(f"    Has learner: {sta.learner is not None}")
            print(f"    Has _fixed_action: {hasattr(sta, '_fixed_action')}")
            print(f"    Condition: npca_enabled={sta.npca_enabled} and npca_channel={sta.npca_channel is not None} and (learner={sta.learner is not None} or fixed_action={hasattr(sta, '_fixed_action')})")
            
            condition_result = (sta.npca_enabled and sta.npca_channel and 
                               (sta.learner or hasattr(sta, '_fixed_action')))
            print(f"    -> 조건 결과: {condition_result}")
            
            if not getattr(sta, '_opt_active', False):
                option_starts.append(slot)
        
        # STA 스텝
        prev_opt_active = getattr(sta, '_opt_active', False)
        sta.step(slot)
        new_opt_active = getattr(sta, '_opt_active', False)
        
        if not prev_opt_active and new_opt_active:
            print(f"*** 옵션 시작 at slot {slot}")
        
        sta.state = sta.next_state
        
        # 상세 로그 (슬롯 0-20)
        if slot <= 20 or (5 <= slot <= 60 and slot % 5 == 0):
            print(f"Slot {slot:2d}: State={sta.state.name:15} | Primary_busy={primary_busy} | OBSS_busy={obss_busy} | Opt_active={getattr(sta, '_opt_active', False)} | Occupancy={sta.channel_occupancy_time}")
    
    print(f"\n=== 결과 ===")
    print(f"옵션 시작 시점들: {option_starts}")
    print(f"총 결정 로그: {len(decision_log)}")
    print(f"최종 점유 시간: {sta.channel_occupancy_time}")
    
    if decision_log:
        print("결정 로그:")
        for log in decision_log:
            print(f"  Slot {log['slot']}: Action {log['action']}")

if __name__ == "__main__":
    debug_decision_logic()