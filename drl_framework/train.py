import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

from drl_framework.network import ReplayMemory, DQN
from drl_framework.params import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

class SemiMDPLearner:
    def __init__(self, n_observations, n_actions, device, memory_capacity=10000, lr=LR):
        self.device = device
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.steps_done = 0
        
        # DQN 네트워크 초기화
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer 및 Memory 초기화
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)
    
    def select_action(self, state_tensor):
        """Epsilon-greedy action selection for Semi-MDP"""
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randint(0, self.n_actions - 1)
    
    def optimize_model(self):
        """Semi-MDP용 최적화 함수 - 옵션 기반 학습"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).long().unsqueeze(1)
        R_batch = torch.tensor(batch.cum_reward, device=self.device).float()  # 옵션 누적 보상
        tau_batch = torch.tensor(batch.tau, device=self.device).float()       # 옵션 길이(슬롯 수)
        done_batch = torch.tensor(batch.done, device=self.device).float()     # 1.0 if done else 0.0

        # Q(s_t, a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # max_a' Q_target(s', a') for non-terminal only
        non_final_mask = (done_batch == 0)
        
        next_state_values = torch.zeros(len(state_batch), device=self.device)
        if non_final_mask.sum() > 0:  # non-terminal states가 있는 경우에만 처리
            non_final_next_states = torch.stack(
                [s for s, d in zip(batch.next_state, batch.done) if not d]
            ).to(self.device)
            
            with torch.no_grad():
                if non_final_next_states.numel() > 0:
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Semi-MDP TD Target: R + γ^τ * V(s') for non-terminal states
        # tau 클리핑으로 극도로 작은 할인 인수 방지
        tau_clipped = torch.clamp(tau_batch, max=20.0)  # 최대 20 슬롯으로 제한
        td_target = R_batch + (GAMMA ** tau_clipped) * next_state_values * (1.0 - done_batch)
        loss = F.smooth_l1_loss(q_sa, td_target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                         target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

def select_action(state, policy_net, env, steps_done, device):
    """Epsilon-greedy action selection"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
    #                               device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch  = torch.stack(batch.state).to(device)                # [B, obs_dim]
    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    R_batch      = torch.tensor(batch.cum_reward, device=device).float()  # 옵션 누적 보상
    tau_batch    = torch.tensor(batch.tau, device=device).float()         # 옵션 길이(슬롯 수)
    done_batch   = torch.tensor(batch.done, device=device).float()         # 1.0 if done else 0.0

    # Q(s_t, a)
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    q_sa = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # # V(s_{t+1})
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # max_a' Q_target(s', a') for non-terminal only
    non_final_mask = (done_batch == 0)
    non_final_next_states = torch.stack(
        [s for s, d in zip(batch.next_state, batch.done) if not d]
    ).to(device)

    next_state_values = torch.zeros(len(state_batch), device=device)
    with torch.no_grad():
        if non_final_next_states.numel() > 0:
            # (DDQN 원하면 여기서 policy_net.argmax로 a' 뽑아 target_net.gather 사용)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Loss
    # criterion = nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    td_target = R_batch + (GAMMA ** tau_batch) * next_state_values * (1.0 - done_batch)
    loss = F.smooth_l1_loss(q_sa, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(env, policy_net, target_net, optimizer, device, num_episodes=50):
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(env.flatten_dict_values(state),
                             dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        for t in count():
            action = select_action(state, policy_net, env, steps_done, device)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            total_reward += reward.item()
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(env.flatten_dict_values(observation),
                                          dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(policy_net, target_net, memory, optimizer, device)

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                             target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                print(f"Episode {i_episode}: total reward = {total_reward}")
                break

    return episode_rewards

def train_semi_mdp(channels, stas_config, num_episodes=100, num_slots_per_episode=1000, device=None):
    """
    Semi-MDP를 사용한 NPCA STA 학습 함수
    
    Args:
        channels: 채널 리스트
        stas_config: STA 설정 리스트
        num_episodes: 학습 에피소드 수
        num_slots_per_episode: 에피소드당 슬롯 수
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SemiMDPLearner 초기화
    n_observations = 4  # obs feature 개수
    n_actions = 2       # 0=StayPrimary, 1=GoNPCA
    learner = SemiMDPLearner(n_observations, n_actions, device)
    
    episode_rewards = []
    episode_losses = []
    
    print(f"Starting Semi-MDP training on {device}")
    print(f"Episodes: {num_episodes}, Slots per episode: {num_slots_per_episode}")
    
    for episode in range(num_episodes):
        # 채널 상태 초기화
        for ch in channels:
            ch.intra_occupied = False
            ch.intra_end_slot = 0
            ch.obss_traffic = []
            ch.occupied_remain = 0
            ch.obss_remain = 0
        
        # STA 생성 및 초기화 (각 에피소드마다 새로 생성)  
        from drl_framework.random_access import STA, Simulator
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
                learner=learner if config.get("npca_enabled", False) else None
            )
            stas.append(sta)
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
        simulator.memory = learner.memory
        simulator.device = device
        simulator.run()
        
        # 에피소드별 총 보상 수집 - episode_reward 사용
        total_reward = 0
        for sta in stas:
            if sta.npca_enabled:  # NPCA 가능한 STA만 보상 수집
                total_reward += sta.episode_reward
                
            # 에피소드 종료 시 남은 옵션들 정리
            if sta._opt_active:
                sta._end_option()
                
            # pending 옵션들을 done=True로 finalize 
            if sta._pending:
                final_obs = sta.get_obs()
                final_obs_vec = sta.obs_to_vec(final_obs, normalize=True)
                sta._finalize_pending_with_next_state(
                    next_obs_vec=final_obs_vec,
                    memory=learner.memory,
                    done=True,  # 에피소드 종료
                    device=learner.device
                )
                
            # 에피소드 보상 초기화
            sta.episode_reward = 0.0
        episode_rewards.append(total_reward)
        
        # 학습 수행 - 빈도 조절
        if len(learner.memory) >= BATCH_SIZE:
            # 에피소드당 1번만 학습으로 안정성 향상
            loss = learner.optimize_model()
            if loss is not None:
                episode_losses.append(loss)
            
            # Target network를 10 에피소드마다 업데이트
            if episode % 10 == 0:
                learner.update_target_network()
        
        # 진행 상황 출력
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_loss = sum(episode_losses[-10:]) / max(1, len(episode_losses[-10:]))
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * learner.steps_done / EPS_DECAY)
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:6.2f}, "
                  f"Avg Loss = {avg_loss:.4f}, Epsilon = {epsilon:.3f}, "
                  f"Memory Size = {len(learner.memory)}")
    
    print("Training completed!")
    return episode_rewards, episode_losses, learner
