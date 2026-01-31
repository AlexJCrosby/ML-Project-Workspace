import numpy as np
import matplotlib.pyplot as plt
from env import MazeEnv

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
    max_steps_per_episode: int = 100,
    alpha: float = 0.1, # learning rate
    gamma: float = 0.99, # discount factor
    epsilon_start: float = 1.0, # initial exploration rate
    epsilon_min: float = 0.01, # minimum exploration rate
    epsilon_decay: float = 0.995, # exploration decay rate
    seed: int = 0
):
    env = MazeEnv()
    rng = np.random.default_rng(seed)
    
    # Q-table: rows = states, columns = actions
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        
        for t in range(max_steps_per_episode):
            # 1. Choose action (epsilon-greedy)
            action = choose_action(state, Q, epsilon, env.n_actions, rng)
            
            # 2. Take step in environment
            next_state, reward, done, _ = env.step(action) 

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
                break
            
        # 4. Decay epsilon after each episode (reduce randomness over time)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)
        
        # Logging: print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            last_100_avg = np.mean(episode_rewards[-100:])
            print( # diagnostic (visualise learning progress)
                f"Episode {episode + 1:4d} | "
                f"Avg Reward (last 100): {last_100_avg:6.2f} | "
                f"Epsilon: {epsilon:5.3f}"
            )
    return Q, episode_rewards

def plot_learning_curve(
    rewards: list[float],
    window: int = 100,
    filename: str = "learning_curve.png"
) -> None:
    """
    Saves a learning curve plot (episode reward + moving average) to a PNG file.
    """
    import matplotlib.pyplot as plt  # imported here to keep the rest of the file lightweight

    if len(rewards) == 0:
        print("No rewards to plot.")
        return

    episodes = np.arange(1, len(rewards) + 1)

    plt.figure()
    plt.plot(episodes, rewards, label="Episode reward")

    # Moving average (only if we have enough episodes)
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        moving_avg = np.convolve(rewards, kernel, mode="valid")
        ma_episodes = np.arange(window, len(rewards) + 1)
        plt.plot(ma_episodes, moving_avg, label=f"Moving average ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-Learning: Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved learning curve plot to: {filename}")

# No learning, just demonstrate the learned policy
def run_greedy_policy(Q: np.ndarray, render: bool = True, max_steps: int = 50):
    """
    Run one episode using the greedy policy (no exploration)
    to see how well the agent has learned.
    """
    env = MazeEnv()
    state = env.reset()
    
    if render:
        env.render()
        
    total_reward = 0.0
    
    for t in range(max_steps):
        action = int(np.argmax(Q[state]))  # always pick best action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if render:
            print(f"Step {t}: action={action}, reward={reward}, done={done}")
            total_reward += reward
            
        state = next_state
        
        if done:
            print(f"Reached goal in {t+1} steps with total reward {total_reward:.2f}.")
            break
    else:
        print(f"Did not reach goal within {max_steps} steps. Total reward: {total_reward:.2f}.")   
        
if __name__ == "__main__":
    # Train Q-learning agent
    Q, rewards = train_q_learning()
        
    # Save a learning curve plot
    plot_learning_curve(rewards, window=100, filename="learning_curve.png")

    # Test the learned policy
    print("\nRunning greedy policy after training:\n")
    run_greedy_policy(Q, render=True)