import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from duke_env import DukeSurvivalEnv

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

def greedy_q_policy(Q: np.ndarray):
    """
    Returns a policy that selects greedy actions from a Q-table.

    Supports both:
      - phase-aware Q (rows == env.n_states)
      - phase-stripped Q (rows == env.n_states // 5), where state must be collapsed via state//5
    """
    def policy(env: DukeSurvivalEnv, state: int, info: dict) -> int:
        s = int(state)

        # If Q-table is smaller than the env's state space, assume it is phase-stripped
        # and collapse: state = tile_index * 5 + phase  -> tile_index
        if Q.shape[0] != getattr(env, "n_states", Q.shape[0]) and Q.shape[0] == getattr(env, "n_states", 0) // 5:
            s = s // 5

        # Extra safety: if still out of bounds, try collapsing once
        if s < 0 or s >= Q.shape[0]:
            s = (int(state) // 5)
        if s < 0 or s >= Q.shape[0]:
            raise IndexError(f"State {state} (mapped to {s}) out of bounds for Q with {Q.shape[0]} states.")

        return int(np.argmax(Q[s]))
    return policy


def random_policy(env: DukeSurvivalEnv, state: int, info: dict) -> int:
    """A simple baseline policy for debugging playback."""
    return int(env.rng.integers(0, env.n_actions))


def run_episode_and_record(env: DukeSurvivalEnv, policy_fn, max_steps: int = 500):
    """
    Runs a single episode and records frames.

    Each frame stores:
      - grid tiles
      - agent position
      - tick, hp, gaze flags
      - action taken (for the transition into this frame)
      - info dict returned by env.step
      - hazard masks (for visualization)
    """
    frames = []

    state = env.reset()
    done = False

    # Record an initial frame BEFORE any action (action=None)
    frames.append(capture_frame(env, action=None, reward=None, done=False, info={"tick": env.t}))

    steps = 0
    while not done and steps < max_steps:
        # You can swap in other policies later (e.g., greedy Q-table)
        action = policy_fn(env, state, {})  # info isn't needed for random policy

        next_state, reward, done, info = env.step(action)

        frames.append(capture_frame(env, action=action, reward=reward, done=done, info=info))

        state = next_state
        steps += 1
    return frames


def capture_frame(env: DukeSurvivalEnv, action, reward, done: bool, info: dict):
    """
    Creates a snapshot for visualization.

    NOTE: This reads env internals (grid, hp, t, etc.) which is fine for debugging tooling.
    """
    grid = np.array(env.grid, copy=True)
    agent_pos = tuple(int(x) for x in env.agent_pos)

    tick = int(info.get("tick", env.t))
    hp = int(info.get("hp", env.hp))
    gaze_active = bool(info.get("gaze_active", env.gaze_active))
    gaze_timer = int(info.get("gaze_timer", env.gaze_timer))


    # With 1-indexed ticks:
    # - telegraph/attack-cycle ticks are 5,10,15,...  => tick % 5 == 0 and tick >= 5
    # - slam ticks are the following ticks: 6,11,16,... => (tick - 1) % 5 == 0 and tick >= 6
    slam_tick = bool(info.get("boss_slam", False))
    telegraph_tick = (tick >= 5) and (tick % 5 == 0)


    # Slam hazard is row=2, cols 3..10 inclusive (1x1 AoE per tile)
    slam_zone = np.zeros_like(grid, dtype=bool)
    slam_zone[2, 3:11] = True  # cols 3..10 inclusive

    slam_hazard_mask = slam_zone & slam_tick

    # Gaze resolution happens when gaze_timer counts down to 0 inside step().
    # In your env, the lethal check occurs when gaze_active and gaze_timer reaches 0.
    # So a useful visual is: "if gaze_active, show pillar tiles as safe targets"
    pillar_mask = (grid == env.TILE_PILLAR)
    gaze_imminent = gaze_active and (gaze_timer <= 1)

    # Create a simple "danger" mask for the viewer:
    # - highlight melee tiles on slam ticks
    # - optionally, if gaze is imminent, highlight non-pillar tiles as danger (but that's "global")
    danger_mask = slam_hazard_mask.copy()

    return {
        "grid": grid,
        "agent_pos": agent_pos,
        "tick": tick,
        "hp": hp,
        "gaze_active": gaze_active,
        "gaze_timer": gaze_timer,
        "telegraph_tick": telegraph_tick,
        "slam_tick": slam_tick,
        "gaze_imminent": gaze_imminent,
        "pillar_mask": pillar_mask,
        "danger_mask": danger_mask,
        "action": action,
        "reward": reward,
        "done": done,
        "info": info,
    }


def play_frames(frames):
    """
    Opens a matplotlib window with arrow-key scrubbing.

    Controls:
      - Right arrow: next frame
      - Left arrow: previous frame
      - Space: autoplay/pause
      - r: restart (go to frame 0)
      - q or esc: quit
    """
    if not frames:
        print("No frames to display.")
        return

    idx = 0
    autoplay = False
    cumulative_rewards = []
    running = 0.0
    for fr in frames:
        r = fr["reward"]
        if r is not None:
            running += float(r)
        cumulative_rewards.append(running)


    # Prepare a tile visualization: use a discrete colormap
    # We keep it simple: map tile codes to themselves and use a categorical colormap
    # (You can refine this later.)
    grid0 = frames[0]["grid"]
    n_rows, n_cols = grid0.shape
    
    fig = plt.figure(figsize=(10, 7))
    fig.canvas.manager.set_window_title("Duke Playback (←/→ step, space play/pause)")

    # 2-column layout: left text, right grid
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.05, 2.2], wspace=0.05)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_grid = fig.add_subplot(gs[0, 1])

    # Text panel setup
    ax_text.axis("off")
    text_box = ax_text.text(
        0.0, 1.0, "",
        transform=ax_text.transAxes,
        va="top", ha="left",
        fontsize=11,
        family="monospace",
    )

    # Grid panel setup
    grid0 = frames[0]["grid"]
    n_rows, n_cols = grid0.shape

    im = ax_grid.imshow(grid0, interpolation="nearest", cmap="tab20")
    ax_grid.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_grid.grid(which="minor", linewidth=0.5)
    ax_grid.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Persistent overlays (create ONCE, then update)
    agent_dot = ax_grid.scatter(
        [0], [0],
        s=220,
        marker="o",
        c="white",
        edgecolors="black",
        linewidths=2.5,
        zorder=10,
    )

    danger_overlay = ax_grid.imshow(
        np.zeros_like(grid0, dtype=float),
        interpolation="nearest",
        cmap="Reds",
        alpha=0.0,
        vmin=0.0,
        vmax=1.0,
    )

    def update_view():
        nonlocal idx

        frame = frames[idx]
        grid = frame["grid"]

        im.set_data(grid)

        # Update agent marker (x=col, y=row)
        r, c = frame["agent_pos"]
        agent_dot.set_offsets([[c, r]])

        # Update danger overlay
        danger = frame["danger_mask"].astype(float)
        danger_overlay.set_data(danger)
        danger_overlay.set_alpha(0.35 if danger.max() > 0 else 0.0)

        # HUD text
        action = frame["action"]
        action_str = "START" if action is None else f"{action} ({ACTION_NAMES.get(action, '?')})"
        reward = frame["reward"]
        reward_str = "-" if reward is None else str(reward)

        info = frame.get("info", {})
        took_slam = info.get("took_slam_damage", False)
        took_gaze = info.get("took_gaze_damage", False)

        action = frame["action"]
        action_str = "START" if action is None else f"{action} ({ACTION_NAMES.get(action, '?')})"
        reward = frame["reward"]
        reward_str = "-" if reward is None else str(reward)

        info = frame.get("info", {})
        took_slam = info.get("took_slam_damage", False)
        took_gaze = info.get("took_gaze_damage", False)

        lines = [
            "STATE",
            f"hp:    {frame['hp']}",
            f"reward total: {cumulative_rewards[idx]:.1f}",
            f"tick:  {frame['tick']}",
            "",
            "< MAGIC >",
            f"magic_hit: {info.get('took_magic_damage', False)}",
            "",
            "< MELEE >",
            f"rise_hit: {info.get('took_rise_damage', False)}",
            f"slam_active: {frame.get('slam_tick', False)}",
            f"slam_hit: {took_slam}",
            "",
            "< GAZE >",
            f"gaze_active:{frame.get('gaze_active', False)}",
            f"gaze_hit: {took_gaze}",
            f"gaze_timer: {frame.get('gaze_timer', 0)}",
            "",
        ]

        text_box.set_text("\n".join(lines))


        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx, autoplay

        if event.key == "right":
            idx = min(idx + 1, len(frames) - 1)
            update_view()
        elif event.key == "left":
            idx = max(idx - 1, 0)
            update_view()
        elif event.key == " ":
            autoplay = not autoplay
        elif event.key == "r":
            idx = 0
            update_view()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Basic autoplay loop using a timer
    # Basic autoplay loop using a SINGLE repeating timer (do not create timers inside the callback)
    def on_timer():
        nonlocal idx, autoplay
        if not plt.fignum_exists(fig.number):
            return

        if autoplay:
            if idx < len(frames) - 1:
                idx += 1
                update_view()
            else:
                autoplay = False


    update_view()

    timer = fig.canvas.new_timer(interval=120)
    timer.add_callback(on_timer)
    timer.start()

    plt.show()

def play_frames_side_by_side(frames_a, frames_b, label_a="A", label_b="B"):
    """
    Side-by-side playback of two frame sequences with synced controls.

    Controls:
      - Right arrow: next frame (both)
      - Left arrow: previous frame (both)
      - Space: autoplay/pause
      - r: restart
      - q or esc: quit
    """
    if not frames_a or not frames_b:
        print("Need two non-empty frame lists.")
        return

    # Clamp helper: if one episode is shorter, hold its last frame.
    def get_frame(frames, idx):
        if idx < 0:
            return frames[0]
        if idx >= len(frames):
            return frames[-1]
        return frames[idx]

    idx = 0
    autoplay = False
    max_len = max(len(frames_a), len(frames_b))

    grid0 = frames_a[0]["grid"]
    n_rows, n_cols = grid0.shape

    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title("Duke Side-by-Side Playback (←/→ step, space play/pause)")
    gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[1.05, 2.2, 1.05, 2.2], wspace=0.05)

    # Left text + grid
    ax_text_a = fig.add_subplot(gs[0, 0]); ax_text_a.axis("off")
    ax_grid_a = fig.add_subplot(gs[0, 1])

    # Right text + grid
    ax_text_b = fig.add_subplot(gs[0, 2]); ax_text_b.axis("off")
    ax_grid_b = fig.add_subplot(gs[0, 3])

    text_a = ax_text_a.text(0.0, 1.0, "", transform=ax_text_a.transAxes,
                            va="top", ha="left", fontsize=11, family="monospace")
    text_b = ax_text_b.text(0.0, 1.0, "", transform=ax_text_b.transAxes,
                            va="top", ha="left", fontsize=11, family="monospace")

    # Shared colormap style
    im_a = ax_grid_a.imshow(grid0, interpolation="nearest", cmap="tab20")
    im_b = ax_grid_b.imshow(grid0, interpolation="nearest", cmap="tab20")

    for ax in (ax_grid_a, ax_grid_b):
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    agent_a = ax_grid_a.scatter([0], [0], s=220, marker="o", c="white",
                                edgecolors="black", linewidths=2.5, zorder=10)
    agent_b = ax_grid_b.scatter([0], [0], s=220, marker="o", c="white",
                                edgecolors="black", linewidths=2.5, zorder=10)

    danger_a = ax_grid_a.imshow(np.zeros_like(grid0, dtype=float), interpolation="nearest",
                                cmap="Reds", alpha=0.0, vmin=0.0, vmax=1.0)
    danger_b = ax_grid_b.imshow(np.zeros_like(grid0, dtype=float), interpolation="nearest",
                                cmap="Reds", alpha=0.0, vmin=0.0, vmax=1.0)

    def hud_lines(frame, label):
        info = frame.get("info", {})
        took_slam = info.get("took_slam_damage", False)
        took_gaze = info.get("took_gaze_damage", False)

        action = frame.get("action", None)
        action_str = "START" if action is None else f"{action} ({ACTION_NAMES.get(action, '?')})"

        return [
            f"[ {label} ]",
            f"hp:    {frame.get('hp', 0)}",
            f"tick:  {frame.get('tick', 0)}",
            f"action:{action_str}",
            "",
            "< MAGIC >",
            f"magic_hit: {info.get('took_magic_damage', False)}",
            "",
            "< MELEE >",
            f"rise_hit: {info.get('took_rise_damage', False)}",
            f"slam_active: {frame.get('slam_tick', False)}",
            f"slam_hit: {took_slam}",
            "",
            "< GAZE >",
            f"gaze_active:{frame.get('gaze_active', False)}",
            f"gaze_hit: {took_gaze}",
            f"gaze_timer: {frame.get('gaze_timer', 0)}",
            "",
        ]

    def update_view():
        nonlocal idx

        fa = get_frame(frames_a, idx)
        fb = get_frame(frames_b, idx)

        im_a.set_data(fa["grid"])
        im_b.set_data(fb["grid"])

        ra, ca = fa["agent_pos"]
        rb, cb = fb["agent_pos"]
        agent_a.set_offsets([[ca, ra]])
        agent_b.set_offsets([[cb, rb]])

        # Danger overlays are optional (recordings from training may not include masks)
        da = np.asarray(fa.get("danger_mask", np.zeros_like(fa["grid"], dtype=float)), dtype=float)
        db = np.asarray(fb.get("danger_mask", np.zeros_like(fb["grid"], dtype=float)), dtype=float)

        danger_a.set_data(da)
        danger_b.set_data(db)

        danger_a.set_alpha(0.35 if da.size and da.max() > 0 else 0.0)
        danger_b.set_alpha(0.35 if db.size and db.max() > 0 else 0.0)

        text_a.set_text("\n".join(hud_lines(fa, label_a)))
        text_b.set_text("\n".join(hud_lines(fb, label_b)))

        fig.suptitle(f"Frame {idx+1}/{max_len}  (A={label_a}, B={label_b})")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx, autoplay
        if event.key == "right":
            idx = min(idx + 1, max_len - 1)
            update_view()
        elif event.key == "left":
            idx = max(idx - 1, 0)
            update_view()
        elif event.key == " ":
            autoplay = not autoplay
        elif event.key == "r":
            idx = 0
            update_view()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def on_timer():
        nonlocal idx, autoplay
        if not plt.fignum_exists(fig.number):
            return
        if autoplay:
            if idx < max_len - 1:
                idx += 1
                update_view()
            else:
                autoplay = False

    update_view()
    timer = fig.canvas.new_timer(interval=120)
    timer.add_callback(on_timer)
    timer.start()
    plt.show()


def load_recording(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    # Choose which two recordings to compare
    left_path  = os.path.join("recordings", "phase_ep0001.pkl")
    right_path = os.path.join("recordings", "phase_ep0500.pkl")

    frames_left = load_recording(left_path)
    frames_right = load_recording(right_path)

    # If your playback already supports side-by-side, call it here:
    play_frames_side_by_side(frames_left, frames_right, label_a="Episode 1", label_b="Episode 500")


if __name__ == "__main__":
    main()
