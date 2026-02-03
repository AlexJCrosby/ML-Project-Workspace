import numpy as np
import matplotlib.pyplot as plt
from duke_env import DukeSurvivalEnv

# action selection
def choose_action(state: int, Q: np.ndarray, epsilon:float, n_actions: int, rng: np.random.Generator) -> int:
    """
    Epsilon-greedy action selection.
    - With probability epsilon, choose a random action (explore)
    - Otherwise, choose the action with the highest Q-value (exploit)
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))  # Explore: random action
    else:
        return int(np.argmax(Q[state]))  # Exploit: best action based on Q-tabler
    
    
# Main Q-learning training loop
def train_q_learning(
    num_episodes: int = 2000,
    max_steps_per_episode: int = 10000,
    alpha: float = 0.1, # learning rate
    gamma: float = 0.99, # discount factor
    epsilon_start: float = 1.0, # initial exploration rate
    epsilon_min: float = 0.01, # minimum exploration rate
    epsilon_decay: float = 0.995, # exploration decay rate
    seed: int = 0
):
    env = DukeSurvivalEnv(max_steps=max_steps_per_episode, seed=seed)
    rng = np.random.default_rng(seed)
    
    # Q-table: rows = states, columns = actions
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_survival_ticks = []
    episode_damage_magic = []
    episode_damage_rise = []
    episode_damage_slam = []
    episode_damage_gaze = []
    episode_total_damage = []
    episode_died = []
    episode_damage_rate = []


    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        survival_ticks = 0
        dmg_magic = 0
        dmg_rise = 0
        dmg_slam = 0
        dmg_gaze = 0
        died = False
        
        for t in range(max_steps_per_episode):
            # 1. Choose action (epsilon-greedy)
            action = choose_action(state, Q, epsilon, env.n_actions, rng)
            
            # 2. Take step in environment
            next_state, reward, done, info = env.step(action)
            survival_ticks += 1

            # Damage breakdown (matches duke_env.py flags)
            if info.get("took_magic_damage", False):
                dmg_magic += 28
            if info.get("took_rise_damage", False):
                dmg_rise += 3
            if info.get("took_slam_damage", False):
                dmg_slam += env.slam_damage
            if info.get("took_gaze_damage", False):
                dmg_gaze += env.gaze_damage


            # 3. Q-learning update
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            
            # Where learning happens: 
            # Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            Q[state, action] = new_value
            
            state = next_state # update where it thinks it is
            total_reward += reward # track how this episode is going
            
            if done:
                died = (env.hp <= 0)
                break
            
        # 4. Decay epsilon after each episode (reduce randomness over time)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)
        episode_survival_ticks.append(survival_ticks)

        episode_damage_magic.append(dmg_magic)
        episode_damage_rise.append(dmg_rise)
        episode_damage_slam.append(dmg_slam)
        episode_damage_gaze.append(dmg_gaze)

        total_dmg = dmg_magic + dmg_rise + dmg_slam + dmg_gaze
        episode_total_damage.append(total_dmg)

        # Damage rate: damage per tick survived
        damage_rate = total_dmg / max(1, survival_ticks)
        episode_damage_rate.append(damage_rate)

        episode_died.append(int(died))


        
        # FIX USING A STATIC VALUE
        # ASSIGN A VARIABLE TO THE VALUE DESIRED
        # Logging: print progress every 25 episodes
        if (episode + 1) % 50 == 0:
            last_50_avg = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward (50): {last_50_avg:6.2f} | "
                f"Avg Survival (50): {np.mean(episode_survival_ticks[-50:]):5.1f} | "
                f"Epsilon: {epsilon:5.3f}"
            )
        if (episode + 1) in (1, 25, 100, 500):
            np.save(f"Q_ep{episode+1:04d}.npy", Q)

    logs = {
        "reward": episode_rewards,
        "survival_ticks": episode_survival_ticks,
        "damage_magic": episode_damage_magic,
        "damage_rise": episode_damage_rise,
        "damage_slam": episode_damage_slam,
        "damage_gaze": episode_damage_gaze,
        "damage_total": episode_total_damage,
        "damage_rate": episode_damage_rate,
        "died": episode_died,
    }
    return Q, logs


def plot_metric_curve(
    values: list[float],
    title: str,
    ylabel: str,
    window: int = 100,
    filename: str = "plot.png"
) -> None:
    if len(values) == 0:
        print(f"No values to plot for {title}.")
        return

    episodes = np.arange(1, len(values) + 1)

    plt.figure()
    plt.plot(episodes, values, label=ylabel)

    if len(values) >= window:
        kernel = np.ones(window) / window
        moving_avg = np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
        ma_episodes = np.arange(window, len(values) + 1)
        plt.plot(ma_episodes, moving_avg, label=f"Moving average ({window})")

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot to: {filename}")

# No learning, just demonstrate the learned policy
def run_greedy_policy(Q: np.ndarray, max_steps: int = 100):
    """
    Run one episode using the greedy policy (no exploration)
    to see how well the agent has learned in the Duke survival env.
    """
    env = DukeSurvivalEnv(max_steps=max_steps, seed=0)
    state = env.reset()

    total_reward = 0.0

    for t in range(max_steps):
        action = int(np.argmax(Q[state]))  # always pick best action
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            died = (env.hp <= 0)
            if died:
                print(f"Died on tick {t+1} | total_reward={total_reward:.2f}")
            else:
                print(f"Episode ended on tick {t+1} | total_reward={total_reward:.2f}")
            break
    else:
        print(f"Max ticks reached ({max_steps}) | total_reward={total_reward:.2f}")

        
if __name__ == "__main__":
    # Train Q-learning agent
    Q, logs = train_q_learning()

    plot_metric_curve(logs["reward"], "Q-Learning: Episode Reward", "Total reward", window=15, filename="reward_curve.png")
    plot_metric_curve(logs["survival_ticks"], "Q-Learning: Survival Time", "Ticks survived", window=15, filename="survival_curve.png")
    plot_metric_curve(logs["damage_rate"], "Q-Learning: Damage Rate", "Damage per tick", window=15, filename="damage_rate_curve.png")

    # Test the learned policy
    print("\nRunning greedy survival policy after training:\n")
    run_greedy_policy(Q, max_steps=5000)