#!/usr/bin/env python3
"""
Always Primary의 채널 점유 시간 디버깅
"""

from drl_framework.random_access import Channel, STA, Simulator

def debug_primary_occupancy():
    """Always Primary의 점유 시간이 0인 이유 분석"""
    print("Always Primary 점유 시간 디버깅...")
    
    # 채널 설정
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # NPCA
        Channel(channel_id=1, obss_generation_rate=0.6, obss_duration_range=(80, 150))  # Primary
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
    
    print(f"초기 설정:")
    print(f"  Action: {sta._fixed_action} (StayPrimary)")
    print(f"  PPDU duration: {sta.ppdu_duration} slots")
    print(f"  Primary channel OBSS rate: {channels[1].obss_generation_rate}")
    
    # 시뮬레이션 실행
    simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
    
    print(f"\n=== 수동 시뮬레이션 (디버깅용) ===")
    # 슬롯별 상세 추적
    tx_attempts = []
    tx_success_slots = []
    obss_busy_slots = []
    
    for slot in range(200):
        # 채널 업데이트
        for ch in channels:
            ch.update(slot)
        
        # Primary 채널 상태 기록
        if channels[1].is_busy(slot):
            obss_busy_slots.append(slot)
        
        # 전송 시도 확인
        if sta.state.name == "PRIMARY_TX":
            tx_attempts.append(slot)
            if hasattr(sta, 'tx_success') and sta.tx_success:
                tx_success_slots.append(slot)
        
        # STA 스텝
        sta.step(slot)
        sta.state = sta.next_state
        
        # 처음 20 슬롯 상세 로그
        if slot < 20:
            print(f"Slot {slot:3d}: State={sta.state.name:15} | Primary_busy={channels[1].is_busy(slot)} | Occupancy={sta.channel_occupancy_time:3d} | Backoff={sta.backoff:2d}")
    
    manual_occupancy = sta.channel_occupancy_time
    print(f"수동 시뮬레이션 점유 시간: {manual_occupancy}")
    
    print(f"\n=== Simulator.run() 비교 ===")
    # 새로운 STA로 Simulator.run() 테스트
    sta2 = STA(
        sta_id=0,
        channel_id=1,
        primary_channel=channels[1],
        npca_channel=channels[0],
        npca_enabled=True,
        ppdu_duration=33,
        radio_transition_time=1,
        learner=None
    )
    sta2._fixed_action = 0
    
    simulator2 = Simulator(num_slots=200, channels=channels, stas=[sta2])
    simulator2.run()
    
    print(f"Simulator.run() 후 점유 시간: {sta2.channel_occupancy_time}")
    print(f"Simulator.run() 후 episode_reward: {sta2.episode_reward}")
    
    print(f"\n=== 결과 분석 ===")
    print(f"총 OBSS busy 슬롯: {len(obss_busy_slots)}")
    print(f"총 전송 시도 슬롯: {len(tx_attempts)}")
    print(f"총 성공 전송 슬롯: {len(tx_success_slots)}")
    print(f"최종 점유 시간: {sta.channel_occupancy_time}")
    print(f"점유율: {sta.channel_occupancy_time / 200 * 100:.1f}%")
    
    if tx_attempts:
        print(f"전송 시도 슬롯: {tx_attempts[:10]}...")
    if tx_success_slots:
        print(f"성공 전송 슬롯: {tx_success_slots[:10]}...")
    
    # OBSS 패턴 분석
    if obss_busy_slots:
        print(f"OBSS busy 구간: 슬롯 {obss_busy_slots[0]}-{obss_busy_slots[-1]} 등")
        idle_gaps = []
        prev_slot = -1
        for slot in obss_busy_slots:
            if slot - prev_slot > 1:
                gap_length = slot - prev_slot - 1
                if gap_length > 0:
                    idle_gaps.append(gap_length)
            prev_slot = slot
        print(f"Primary 채널 idle 구간 길이들: {idle_gaps[:10]}")
        
        # 전송 가능한 idle 구간 (33+ 슬롯) 분석
        long_gaps = [gap for gap in idle_gaps if gap >= 33]
        print(f"전송 가능한 긴 idle 구간 ({33}+ 슬롯): {len(long_gaps)}개")
        if long_gaps:
            print(f"  구간 길이들: {long_gaps}")

if __name__ == "__main__":
    debug_primary_occupancy()