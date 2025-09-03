import pandas as pd
import torch
import random
from typing import List

# Copy-pasting the given code components into the environment to enable simulation
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict

# Fix random seed for reproducibility
random.seed(42)

# Constants
CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]  # CW from 15 to 1023
SLOTTIME = 9  # μs

class STAState(Enum):
    PRIMARY_BACKOFF = auto()
    PRIMARY_FROZEN = auto()
    PRIMARY_TX = auto()
    NPCA_BACKOFF = auto()
    NPCA_FROZEN = auto()
    NPCA_TX = auto()

# @dataclass
# class OBSSTraffic:
#     obss_id: str
#     start_slot: int
#     duration: int
#     source_bss: Optional[int] = None
#     @property
#     def end_slot(self):
#         return self.start_slot + self.duration

@dataclass
class OccupyRequest:
    channel_id: int
    duration: int
    is_obss: bool = False

class Channel:
    def __init__(self, channel_id: int, obss_generation_rate: float = 0.0, obss_duration_range: Tuple[int, int] = (20, 40)):
        self.channel_id = channel_id
        self.obss_generation_rate = obss_generation_rate
        self.obss_duration_range = obss_duration_range

        # Intra-BSS 점유 상태
        self.intra_occupied = False
        self.intra_end_slot = 0

        # OBSS 트래픽 리스트: (obss_id, start_slot, duration, source_bss)
        self.obss_traffic: List[Tuple[str, int, int, int]] = []
        
        # 남은 점유시간 캐시 (슬롯마다 update에서 갱신)
        self.occupied_remain = 0        # intra-BSS 점유 남은 시간
        self.obss_remain = 0            # OBSS 점유 남은 시간

    def occupy(self, slot: int, duration: int, sta_id: int):
        """STA가 채널을 점유함 (intra-BSS 점유)"""
        self.intra_occupied = True
        self.intra_end_slot = slot + duration
         # 캐시를 즉시 반영 (옵션이지만 추천)
        self.occupied_remain = duration

    def add_obss_traffic(self, req: OccupyRequest, slot: int):
        """NPCA 전송을 OBSS 트래픽으로 기록"""
        obss_tuple = (
            f"obss_gen_{self.channel_id}_slot{slot}",
            slot,
            req.duration,
            req.source_bss if hasattr(req, "source_bss") else -1  # fallback
        )
        self.obss_traffic.append(obss_tuple)

    def is_busy_by_intra_bss(self, slot: int) -> bool:
        # return self.intra_occupied and self.intra_end_slot > slot
        return self.occupied_remain > 0  # update()에서 이미 최신화

    def is_busy_by_obss(self, slot: int) -> bool:
        # return any(start <= slot < start + dur for _, start, dur, _ in self.obss_traffic)
        return self.obss_remain > 0

    def is_busy(self, slot: int) -> bool:
        # return self.is_busy_by_intra_bss(slot) or self.is_busy_by_obss(slot)
        return (self.occupied_remain > 0) or (self.obss_remain > 0)

    def update(self, slot: int):
        """슬롯마다 상태 갱신: 점유 만료/OBSS 제거 + 남은 점유시간 캐시 갱신"""
        if self.intra_occupied and self.intra_end_slot <= slot:
            self.intra_occupied = False

        # 유효한 OBSS만 유지
        self.obss_traffic = [t for t in self.obss_traffic if t[1] + t[2] > slot]

        # 🔁 남은 점유시간 갱신
        self.occupied_remain = max(0, self.intra_end_slot - slot) if self.intra_occupied else 0

        # 현재 slot에 활성화된 OBSS가 있다면 그 중 "가장 늦게 끝나는" 남은 시간으로 설정
        # (여러 OBSS가 겹치는 경우를 커버; 단일만 있으면 동일 동작)
        active_obss = [start + dur - slot for _, start, dur, _ in self.obss_traffic if start <= slot < start + dur]
        self.obss_remain = max(active_obss) if active_obss else 0


    def generate_obss(self, slot: int):
        """OBSS 트래픽을 확률적으로 생성"""
        if self.obss_generation_rate == 0:
            return

        if not self.is_busy(slot):
            if random.random() < self.obss_generation_rate:
                duration = random.randint(*self.obss_duration_range)
                obss_tuple = (
                    f"obss_gen_{self.channel_id}_slot{slot}",
                    slot,
                    duration,
                    -1  # source_bss unknown
                )
                self.obss_traffic.append(obss_tuple)
                
    def get_latest_obss(self, slot: int) -> Optional[Tuple[str, int, int, int]]:
        """현재 slot에 유효한 OBSS 중 가장 최근에 시작된 것을 반환"""
        active = [
            obss for obss in self.obss_traffic
            if obss[1] <= slot < obss[1] + obss[2]  # start <= slot < end
        ]
        if not active:
            return None
        return max(active, key=lambda x: x[1])  # start_slot 기준으로 가장 최근



class STA:
    def __init__(self, 
                 sta_id: int, 
                 channel_id: int, 
                 primary_channel: Channel, 
                 npca_channel: Optional[Channel] = None, 
                 npca_enabled: bool = False, 
                 radio_transition_time: int = 1,
                 ppdu_duration: int = 10):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.primary_channel = primary_channel
        self.npca_channel = npca_channel
        self.npca_enabled = npca_enabled
        self.radio_transition_time = radio_transition_time
        self.occupy_request: Optional[OccupyRequest] = None

        self.state = STAState.PRIMARY_BACKOFF
        self.next_state = self.state
        self.cw_index = 0
        self.backoff = self.generate_backoff() + 1
        self.tx_remaining = 0
        self.ppdu_duration = ppdu_duration
        # self.current_obss: Optional[OBSSTraffic] = None
        self.intent = None

        # 옵션 관련 변수 초기화
        self._opt_active = False
        self._opt_s = None          # dict (관측 원본; 나중에 벡터화)
        self._opt_a = None          # int (0=StayPrimary, 1=GoNPCA 등)
        self._opt_R = 0.0           # 누적 보상 (슬롯 합산)
        self._opt_tau = 0           # 옵션 sojourn length (슬롯 수)
        self._pending = None        # (s_dict, a, cum_R, tau) — 다음 결정 때 s' 채워 push

    def generate_backoff(self) -> int:
        cw = CONTENTION_WINDOW[self.cw_index]
        return random.randint(0, cw)
    
    def handle_collision(self):
        self.cw_index = min(self.cw_index + 1, len(CONTENTION_WINDOW) - 1)
        self.backoff = self.generate_backoff()
        self.tx_remaining = 0
        self.next_state = STAState.PRIMARY_BACKOFF

    def handle_success(self):
        self.cw_index = 0
        self.backoff = self.generate_backoff()
        self.next_state = STAState.PRIMARY_BACKOFF
    
    def decide_action(self, slot):
        self.intent = None
        if self.state == STAState.PRIMARY_BACKOFF and self.backoff == 0:
            self.intent = "primary_tx"
        return self.intent

    def get_tx_duration(self, is_npca=False) -> int:
        if is_npca:
            return min(self.primary_channel.obss_remain, self.ppdu_duration)
        return self.ppdu_duration
    
    def get_obs(self):
        obs = {
            "primary_channel_obss_occupied_remained": self.primary_channel.obss_remain,
            "radio_transition_time": self.radio_transition_time,
            "tx_duration": self.get_tx_duration(),
            "cw_index": self.cw_index,
        }
        return obs
    
    def obs_to_vec(self, obs: dict, normalize: bool = False, caps=None):
        FEATURE_ORDER = (
            "primary_channel_obss_occupied_remained",
            "radio_transition_time",
            "tx_duration",
            "cw_index",
        )
        x = [float(obs[k]) for k in FEATURE_ORDER]
        if not normalize:
            return x
        caps = caps or {"slots": 1024, "cw_stage_max": 8}
        x[0] = min(x[0], caps["slots"]) / caps["slots"]
        x[1] = min(x[1], caps["slots"]) / caps["slots"]
        x[2] = min(x[2], caps["slots"]) / caps["slots"]
        x[3] = min(x[3], caps["cw_stage_max"]) / caps["cw_stage_max"]
        return x

    def step(self, slot: int):
        if self.state == STAState.PRIMARY_BACKOFF:
            self._handle_primary_backoff(slot)
        elif self.state == STAState.PRIMARY_FROZEN:
            self._handle_primary_frozen(slot)
        elif self.state == STAState.PRIMARY_TX:
            self._handle_primary_tx(slot)
        elif self.state == STAState.NPCA_BACKOFF:
            self._handle_npca_backoff(slot)
        elif self.state == STAState.NPCA_FROZEN:
            self._handle_npca_frozen(slot)
        elif self.state == STAState.NPCA_TX:
            self._handle_npca_tx(slot)

    def _handle_primary_backoff(self, slot: int):
        # 1. Primary 채널이 intra-BSS busy: frozen
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.PRIMARY_FROZEN
        # 2. Primary 채널이 OBSS busy: NPCA enabled 여부에 따라 다름
        elif self.primary_channel.is_busy_by_obss(slot):
            # NPCA enabled인 경우
            if self.npca_enabled and self.npca_channel:
                # [결정 시점] 현재 관측
                obs_dict = self.get_obs()
                obs_vec = self.obs_to_vec(obs_dict, normalize=True)

                # 직전 옵션이 끝나 pending이 있다면 지금 관측을 s'로 붙여 push
                self._finalize_pending_with_next_state(
                    next_obs_vec=obs_vec,
                    memory=self.learner.memory,   # 또는 시뮬레이터에서 주입한 메모리
                    done=False,
                    device=self.learner.device
                )

                # 액션 선택 (결정 시점 발생 횟수로만 epsilon 증가)
                action = self.policy.select_action(
                    torch.tensor(obs_vec, dtype=torch.float32, device=self.learner.device).unsqueeze(0)
                )
                self.learner.steps_done += 1

                # 옵션 시작 (이번 (s,a) 기록)
                self._begin_option(obs_dict, int(action))

                # 기존 분기 유지
                self.current_obss = self.primary_channel.get_latest_obss(slot)
                self.cw_index = 0
                self.backoff = self.generate_backoff()

                if action == 0:
                    self.next_state = STAState.PRIMARY_FROZEN
                else:
                    if self.npca_channel.is_busy_by_intra_bss(slot):
                        self.next_state = STAState.NPCA_FROZEN
                    # NPCA 채널이 busy하지 않으면 backoff
                    else:
                        self.next_state = STAState.NPCA_BACKOFF
            else:
                self.next_state = STAState.PRIMARY_FROZEN
        # 3. Primary 채널이 idle:
        else:
            if (self.backoff == 0) and not self.primary_channel.is_busy(slot):
                self.tx_remaining = self.get_tx_duration()
                self.occupy_request = OccupyRequest(
                    channel_id=self.primary_channel.channel_id, 
                    duration=self.tx_remaining, 
                    is_obss=False)
                self.next_state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1 if self.backoff > 0 else 0
        return
    
    def _handle_primary_frozen(self, slot: int):
        if not self.primary_channel.is_busy(slot):
            self.next_state = STAState.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int):
        # Primary_tx 동안 OBSS 점유 히스토리가 있다면 무조건 tx_success is False
        if self.primary_channel.is_busy_by_obss(slot):
            self.tx_success = False
        else:
            self.tx_success = True

        # 전송 중
        if self.tx_remaining > 0:
            self.tx_remaining -= 1

        # 전송 종료 후
        if self.tx_remaining == 0:
            # self.next_state = STAState.PRIMARY_BACKOFF
            # self.backoff = self.generate_backoff()
            # self.cw_index = 0
            if self.tx_success:
                self.handle_success()  # 전송 성공 처리
            else:
                self.handle_collision()  # 전송 실패 처리
        self._end_option()

    def _handle_npca_backoff(self, slot: int):
        # 1. NPCA 채널이 busy: frozen
        if self.npca_channel.is_busy(slot):
            self.next_state = STAState.NPCA_FROZEN
        # 2. NPCA 채널이 idle: backoff
        else:
            if (self.backoff == 0) and not self.npca_channel.is_busy(slot):
                self.tx_remaining = self.get_tx_duration(is_npca=True)
                self.occupy_request = OccupyRequest(
                    channel_id=self.npca_channel.channel_id,
                    duration=self.tx_remaining,
                    is_obss=True
                )
                self.next_state = STAState.NPCA_TX
            else:
                self.backoff -= 1 if self.backoff > 0 else 0


    def _handle_npca_frozen(self, slot: int):
        # # OBSS 정보가 더 이상 유효하지 않으면 primary로 복귀
        if self.primary_channel.obss_remain == 0:
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            self.next_state = STAState.PRIMARY_BACKOFF

        if not self.npca_channel.is_busy(slot):
            self.next_state = STAState.NPCA_BACKOFF


    def _handle_npca_tx(self, slot: int):
        if self.tx_remaining > 0:
            self.tx_remaining -= 1
            return

        if self.tx_remaining == 0:
            self.current_obss = None  # 전송 종료 → cleanup
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            # If OBSS가 남아있지 않으면,
            if self.primary_channel.obss_remain == 0:
                self.next_state = STAState.PRIMARY_BACKOFF
            # OBSS가 남아있으면,
            else:
                self.next_state = STAState.NPCA_BACKOFF
            self._end_option()
            return
        
    def _begin_option(self, s_dict, a_int):
        assert not self._opt_active, "Option already active"
        self._opt_active = True
        self._opt_s = s_dict
        self._opt_a = int(a_int)
        self._opt_R = 0.0
        self._opt_tau = 0

    def _accum_option_reward(self, r):
        if self._opt_active:
            self._opt_R += float(r)
            self._opt_tau += 1

    def _end_option(self):
        """옵션 종료: (s,a,R,τ)를 pending에 저장. s'는 다음 결정 시 붙임."""
        if self._opt_active:
            self._pending = (self._opt_s, self._opt_a, self._opt_R, self._opt_tau)
            self._opt_active = False
            self._opt_s = None
            self._opt_a = None
            self._opt_R = 0.0
            self._opt_tau = 0

    def _finalize_pending_with_next_state(self, next_s_dict_vec, memory, done: bool, normalize=True, device=None):
        """
        다음 '결정 시점'에서 호출.
        pending 있으면 s'로 next_s_dict_vec를 채워 replay buffer에 push.
        """
        if self._pending is None:
            return
        s_dict, a, R, tau = self._pending
        s_vec  = self.obs_to_vec(s_dict, normalize=normalize)
        s_vec  = torch.tensor(s_vec, dtype=torch.float32, device=device)
        s_next = torch.tensor(next_s_dict_vec, dtype=torch.float32, device=device)
        memory.push(s_vec, a, s_next, R, tau, done)  # (state, action, next_state, cum_reward, tau, done)
        self._pending = None


class Simulator:
    def __init__(self, num_slots: int, stas: List['STA'], channels: List['Channel']):
        self.num_slots = num_slots
        self.stas = stas
        self.channels = channels
        self.log = []

    def run(self):
        for slot in range(self.num_slots):
            # ① 채널 업데이트
            for ch in self.channels:
                ch.update(slot)

            # ② STA 상태 업데이트
            for sta in self.stas:
                sta.occupy_request = None
                sta.step(slot)

            # ③ 채널 OBSS request 수집
            obss_reqs = []
            for ch in self.channels:
                obss_req = ch.generate_obss(slot)
                if obss_req:
                    obss_reqs.append((None, obss_req))

            # ④ STA 전송 요청 수집
            sta_reqs = [(sta, sta.occupy_request) for sta in self.stas if sta.occupy_request is not None]

            # ⑤ 전체 요청 통합
            all_reqs = sta_reqs + obss_reqs

            # ⑥ 채널별로 OccupyRequest 분류
            channel_requests = defaultdict(list)
            for sta, req in all_reqs:
                channel_requests[req.channel_id].append((sta, req))

            # ⑦ Occupy 요청 처리
            for ch_id, reqs in channel_requests.items():
                if len(reqs) == 1:
                    sta, req = reqs[0]
                    if req.is_obss:
                        self.channels[ch_id].add_obss_traffic(req, slot)
                    else:
                        self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                    if sta:
                        sta.tx_success = True
                else:
                    for sta, req in reqs:
                        if sta is not None:
                            if req.is_obss:
                                self.channels[ch_id].add_obss_traffic(req, slot)
                            else:
                                self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                            sta.tx_success = False
                            # sta.handle_collision()

            # ⑧ 상태 전이 및 초기화
            for sta in self.stas:
                sta._accum_option_reward(0)
                sta.state = sta.next_state

            # ⑨ 로그 저장
            self.log_slot(slot)

        for sta in self.stas:
            # 옵션이 살아있다면 종료 -> pending으로 전환
            if sta._opt_active:
                sta._end_option()
            # pending이 있으면 done=True로 push
            if sta._pending is not None:
                s_dict, a, R, tau = sta._pending
                s_vec = torch.tensor(sta.obs_to_vec(s_dict, normalize=True), dtype=torch.float32, device=self.device)
                dummy_next = torch.zeros_like(s_vec)
                self.memory.push(s_vec, a, dummy_next, R, tau, True)  # done=True
                sta._pending = None

    def log_slot(self, slot: int):
        row = {
            "slot": slot,
            "time": slot * SLOTTIME,
        }

        for ch_id, ch in enumerate(self.channels):
            stas_in_ch = [sta for sta in self.stas if sta.channel_id == ch_id]

            row[f"states_ch_{ch_id}"] = [sta.state.name.lower() for sta in stas_in_ch]
            row[f"backoff_ch_{ch_id}"] = [sta.backoff for sta in stas_in_ch]
            row[f"npca_enabled_ch_{ch_id}"] = [sta.npca_enabled for sta in stas_in_ch]

            row[f"channel_{ch_id}_occupied_remained"] = ch.occupied_remain
            row[f"channel_{ch_id}_obss_occupied_remained"] = ch.obss_remain
            
        self.log.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    
