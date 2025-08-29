import os
import pandas as pd
from random_access.configs import simulation_configs
from random_access.random_access import *

save_dir = "./sim_results"
os.makedirs(save_dir, exist_ok=True)

for config in simulation_configs:
    num_slots = config["simulation_time"]
    stas_per_channel = config["stas_per_channel"]
    npca_enabled = config["npca_enabled"]
    frame_size = config["frame_size"]
    obss_rate = config["obss_generation_rate"]
    obss_range = config["obss_frame_size_range"]
    radio_delay = config.get("radio_delay", 0)

    # 채널 생성
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),
        Channel(channel_id=1, obss_generation_rate=obss_rate, obss_duration_range=obss_range)
    ]
    
    # STAs 생성
    stas = []
    sta_id = 0
    for ch_id, num_stas in enumerate(stas_per_channel):
        for _ in range(num_stas):
            sta = STA(
                sta_id=sta_id,
                channel_id=ch_id,
                primary_channel=channels[ch_id],
                npca_channel=channels[0] if ch_id == 1 else None,
                npca_enabled=npca_enabled[ch_id],
                radio_transition_time=radio_delay,
                ppdu_duration=frame_size
            )
            stas.append(sta)
            sta_id += 1
    
    # 시뮬레이터 생성 및 실행
    sim = Simulator(num_slots=num_slots, channels=channels, stas=stas)
    sim.run()
    df = sim.get_dataframe()
    
    # 결과 저장
    df.to_csv(f"{save_dir}/sim_result_{config['label']}.csv", index=False)
    df.to_pickle(f"{save_dir}/sim_result_{config['label']}.pkl")
    print(f"Simulation for {config['label']} completed and saved to {save_dir}/sim_result_{config['label']}.pkl")
