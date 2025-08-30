def train(env, policy_net, target_net, optimizer, device, num_episodes=50, plot_callback=None):
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

            # Soft update
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                             target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                if plot_callback:   # <-- 여기서 실시간 plot 호출
                    plot_callback(episode_rewards)
                break

    return episode_rewards
