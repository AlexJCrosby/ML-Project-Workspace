import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from duke_env import DukeSurvivalEnv

def capture_frame(env, action, reward, done, info):
    """Minimal frame snapshot for playback; reads env internals."""
    return {
        "agent_pos": tuple(int(x) for x in env.agent_pos),
        "tick": int(info.get("tick", env.t)),
        "hp": int(info.get("hp", env.hp)),
        "gaze_active": bool(info.get("gaze_active", getattr(env, "gaze_active", False))),
        "gaze_timer": int(info.get("gaze_timer", getattr(env, "gaze_timer", 0))),
        "action": action,
        "reward": reward,
        "done": bool(done),
        "info": info,
        # Optional but very useful for visuals/debug:
        "grid": np.array(env.grid, copy=True),
    }

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
    

def strip_phase(state: int) -> int:
    """
    Remove tick phase from state.
    Assumes state = tile_index * 5 + phase.
    """
    return state // 5

# Main Q-learning training loop
def train_q_learning(
    num_episodes: int = 1200,
    max_steps_per_episode: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    step_reward: float = 1.0,
    death_penalty: float = -100.0,
    damage_scale: float = 0.1,
    seed: int = 0,
    use_phase: bool = True,
    record_episodes: tuple[int, ...] = (1, 500),
    recordings_dir: str = "recordings",
):

    env = DukeSurvivalEnv(
    max_steps=max_steps_per_episode,
    seed=seed,
    step_reward=step_reward,
    death_penalty=death_penalty,
    damage_scale=damage_scale,
)
    rng = np.random.default_rng(seed)
    
    os.makedirs(recordings_dir, exist_ok=True)
    
    # Q-table: rows = states, columns = actions
    n_states = env.n_states if use_phase else (env.n_states // 5)
    Q = np.zeros((n_states, env.n_actions), dtype=float)
    

    
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
        ep_num = episode + 1
        do_record = ep_num in record_episodes
        frames = []
        if do_record:
            # initial frame before any action
            frames.append(capture_frame(env, action=None, reward=None, done=False, info={"tick": env.t, "hp": env.hp}))
        if not use_phase:
            state = strip_phase(state)
        total_reward = 0.0
        survival_ticks = 0
        dmg_magic = 0
        dmg_rise = 0
        dmg_slam = 0
        dmg_gaze = 0
        died = False
        
        for t in range(max_steps_per_episode):
            action = choose_action(state, Q, epsilon, env.n_actions, rng)

            next_state, reward, done, info = env.step(action)
            if not use_phase:
                next_state = strip_phase(next_state)

            if do_record:
                frames.append(capture_frame(env, action=action, reward=reward, done=done, info=info))

            survival_ticks += 1
            # ... (damage + Q update exactly as you have it)

            state = next_state
            total_reward += reward

            if done:
                died = (env.hp <= 0)
                break
        else:
            # If loop ends without 'break', episode ended by max steps
            died = (env.hp <= 0)

        # Save recording ONCE per episode, after the loop
        if do_record:
            fname = os.path.join(
                recordings_dir,
                f"{'phase' if use_phase else 'baseline'}_ep{ep_num:04d}.pkl"
            )
            with open(fname, "wb") as f:
                pickle.dump(frames, f)
            print(f"Saved episode recording: {fname} ({len(frames)} frames)")


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
        if (episode + 1) % 100 == 0:
            last_100_avg = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward (100): {last_100_avg:6.2f} | "
                f"Avg Survival (100): {np.mean(episode_survival_ticks[-100:]):5.1f} | "
                f"Epsilon: {epsilon:5.3f}"
            )
        # Save Q-table at specific episodes
        if (episode + 1) in (1, 5, 25, 500):
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
def run_greedy_policy(Q: np.ndarray, max_steps: int = 20000, use_phase: bool = True):
    """
    Run one episode using the greedy policy (no exploration)
    to see how well the agent has learned in the Duke survival env.
    """
    env = DukeSurvivalEnv(max_steps=max_steps, seed=0)
    state = env.reset()
    if not use_phase:
        state = strip_phase(state)

    total_reward = 0.0

    for t in range(max_steps):
        action = int(np.argmax(Q[state]))  # always pick best action
        next_state, reward, done, info = env.step(action)
        if not use_phase:
            next_state = strip_phase(next_state)


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
    # Choose which experiment to run
    # "baseline" = ignores phase (state compressed)
    # "phase"    = uses full state (position + tick_mod_5)
    RUN_MODE = "phase"   # <-- change to "baseline" when needed

    if RUN_MODE == "baseline":
        print("\n=== BASELINE RUN (no phase) ===\n")
        use_phase = False
        tag = "baseline"
        title_prefix = "Baseline (No Phase)"

    elif RUN_MODE == "phase":
        print("\n=== PHASE-AWARE RUN ===\n")
        use_phase = True
        tag = "phase"
        title_prefix = "Phase-Aware"
    else:
        raise ValueError('RUN_MODE must be "baseline" or "phase"')

    # Train Q-learning agent
    Q, logs = train_q_learning(use_phase=use_phase)

    # Plots (saved with tag so they don't overwrite each other)
    plot_metric_curve(logs["reward"], f"Q-Learning: Episode Reward ({title_prefix})", "Total reward", window=15, filename=f"reward_curve_{tag}.png")
    plot_metric_curve(logs["survival_ticks"], f"Q-Learning: Survival Time ({title_prefix})", "Ticks survived", window=15, filename=f"survival_curve_{tag}.png")
    plot_metric_curve(logs["damage_rate"], f"Q-Learning: Damage Rate ({title_prefix})", "Damage per tick", window=15, filename=f"damage_rate_curve_{tag}.png")

    # Greedy evaluation
    print(f"\nRunning greedy survival policy ({tag}) after training:\n")
    run_greedy_policy(Q, max_steps=20000, use_phase=use_phase)
