import numpy as np

# Action mapping: int -> (row_change, col_change)
ACTION_DELTAS = {
    0: np.array([-1, 0]),  # Up
    1: np.array([0, 1]),   # Right
    2: np.array([1, 0]),   # Down
    3: np.array([0, -1])   # Left
}

class DukeSurvivalEnv:
    """
    Minimal survival-only environment inspired by Duke Sucellus.

    Goal (for now):
    - Survive as long as possible.
    - No boss HP, no player attacks, no loot, no enrage, no gas flare (optional later).

    Mechanics implemented:
    - Grid with walls (X), boss tiles (1), melee-range tiles (2), vents (3), open tiles (0).
    - A simple repeating 5-tick cadence:
        tick % 5 == 0 -> icicles telegraph (melee tiles become "warned")
        tick % 5 == 1 -> slam lands (standing on melee tile causes damage)
      (This is deliberately simplified but learnable.)
    - Freezing Gaze every 5th "attack event" (i.e., every 25 ticks):
        When it triggers, after 5 ticks you must be on a "pillar zone" (safe columns),
        otherwise you take lethal damage.
      (Also simplified, but gives a second cadence to learn.)

    Reward:
    - +1 per tick alive
    - -100 on death
    """

    # Tile codes
    TILE_WALL = 9
    TILE_OPEN = 0
    TILE_BOSS = 1
    TILE_MELEE = 2
    TILE_VENT = 3
    TILE_PILLAR = 4   # NEW: walkable tiles that protect vs gaze

    def __init__(
        self,
        max_steps: int = 300,
        start_hp: int = 10000,
        slam_damage: int = 20,
        gaze_damage: int = 999,
        step_reward: float = 1.0,
        death_penalty: float = -100.0,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)

        # Your arena layout (interpreting "O" as 0/open)
        # X X 1 1 1 1 1 1 1 X X
        # O O 2 2 2 2 2 2 2 O O
        # O O 3 O O 3 O O 3 O O
        # O O O O O O O O O O O
        # X X O O O O O O O X X
        # O O O O O O O O O O O
        # O O 3 O O 3 O O 3 O O
        # O O O O O O O O O O O
        # X X O O O O O O O X X

        # Updated arena layout (your new design)
        # X = 9, 0 = 0, 1 = boss, 2 = melee, 3 = vent, P = 4
        self.grid = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9],
            [9, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 9],
            [9, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 9],
            [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
            [9, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 9],
            [9, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 9],
            [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ], dtype=int)


        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTION_DELTAS)

        # Parameters
        self.max_steps = int(max_steps)
        self.start_hp = int(start_hp)
        self.slam_damage = int(slam_damage)
        self.gaze_damage = int(gaze_damage)
        self.step_reward = float(step_reward)
        self.death_penalty = float(death_penalty)



        # Precompute walkable mask
        self.walkable = (self.grid != self.TILE_WALL) & (self.grid != self.TILE_BOSS)

        # Pick a sensible start tile: open tile near the middle-bottom
        self.start_pos = np.array([2, 3], dtype=int)

        # Internal state (set on reset)
        self.agent_pos = None
        self.hp = None
        self.t = None
        self.attack_event_count = None
        self.gaze_active = None
        self.gaze_timer = None
        self.pending_slam = None
        self.last_melee_cycle_tick = None
        self.standard_attack_count = 0 



    # --- helpers for state <-> index ---

    def pos_to_state(self, pos: np.ndarray) -> int:
        r, c = int(pos[0]), int(pos[1])
        return r * self.n_cols + c

    def state_to_pos(self, state: int) -> np.ndarray:
        r = state // self.n_cols
        c = state % self.n_cols
        return np.array([r, c], dtype=int)

    def reset(self) -> int:
        self.agent_pos = self.start_pos.copy()
        self.hp = self.start_hp

        # Use 1-indexed ticks to match your design + playback expectations.
        # First attack cycle happens at tick 5.
        self.t = 1

        self.attack_event_count = 0
        self.gaze_active = False
        self.gaze_timer = 0
        self.standard_attack_count = 0

        # Standard attack state
        self.pending_slam = False
        self.last_melee_cycle_tick = None

        return self.pos_to_state(self.agent_pos)


    def _is_pillar_safe(self, pos: np.ndarray) -> bool:
        return self._tile_at(pos) == self.TILE_PILLAR

    def _tile_at(self, pos: np.ndarray) -> int:
        r, c = int(pos[0]), int(pos[1])
        return int(self.grid[r, c])

    def _in_melee_range(self, pos: np.ndarray) -> bool:
        """
        Coordinate-based melee range check (matches your design):
        - y (row) == 2
        - x (col) in [2..10] inclusive
        Includes corner tiles (2,2) and (2,10) even if they are pillar tiles.
        """
        r, c = int(pos[0]), int(pos[1])
        return (r == 2) and (2 <= c <= 10)

    def step(self, action: int):
        """
        Returns: next_state (int), reward (float), done (bool), info (dict)
        """
        assert action in ACTION_DELTAS, f"Invalid action: {action}"

        # --- 1) Move ---
        proposed = self.agent_pos + ACTION_DELTAS[action]
        r, c = int(proposed[0]), int(proposed[1])

        # bounds + walkability
        if (r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols) or (not self.walkable[r, c]):
            # invalid move: stay in place (no extra penalty; survival reward dominates)
            proposed = self.agent_pos.copy()

        self.agent_pos = proposed

        # --- 2) Boss cadence events (simplified) ---
        info = {
            "tick": self.t,
            "hp": self.hp,
            "gaze_active": self.gaze_active,
            "gaze_timer": self.gaze_timer,
            "boss_slam": False,
        }

        # ------------------------------------------------------------
        # Standard attack system (matches your design)
        # - Attack cycle ticks: 5,10,15,...  (first at 5)
        # - On cycle tick:
        #     if in melee range -> Icicle Rise (+3 if in rise AOE), and schedule Slam next tick
        #     else -> Magic hit (28)
        # - On following tick after melee cycle tick -> Icicle Slam (+20 if on slam tiles)
        # ------------------------------------------------------------

        # 1) Resolve Slam ONLY if we scheduled it from a melee cycle
        if (not self.gaze_active) and self.pending_slam and (self.last_melee_cycle_tick is not None) and (self.t == self.last_melee_cycle_tick + 1):
            info["boss_slam"] = True
            
            r, c = int(self.agent_pos[0]), int(self.agent_pos[1])
            in_slam = (r == 2) and (3 <= c <= 10)
            
            if in_slam:
                self.hp -= self.slam_damage
                info["took_slam_damage"] = True
            else:
                info["took_slam_damage"] = False
            # Slam happens once, then clears
            self.pending_slam = False


        # 2) Attack cycle tick (5,10,15,...)
        if self.t % 5 == 0:

            # If gaze is active, it suppresses all standard attacks (including on cycle ticks)
            if self.gaze_active:
                pass

            else:
                # Every 5th would-be standard attack is replaced by starting gaze.
                # That means: after 4 standard attacks, the next cycle tick starts gaze.
                if self.standard_attack_count >= 4:
                    self.gaze_active = True
                    self.gaze_timer = 6  # gaze resolves 5 ticks later
                    self.standard_attack_count = 0

                    # Make absolutely sure no slam can occur due to earlier state
                    self.pending_slam = False
                    self.last_melee_cycle_tick = None

                    # Optional: for playback/debug
                    info["boss_gaze_start"] = True

                else:
                    # Do a standard attack (melee or magic)
                    if self._in_melee_range(self.agent_pos):
                        # Icicle Rise damage if in rise AoE (rows 2..3, cols 2..10)
                        r, c = int(self.agent_pos[0]), int(self.agent_pos[1])
                        in_rise_aoe = (r in (2, 3)) and (2 <= c <= 10)

                        if in_rise_aoe:
                            self.hp -= 3
                            info["took_rise_damage"] = True
                        else:
                            info["took_rise_damage"] = False

                        # Schedule Slam next tick
                        self.pending_slam = True
                        self.last_melee_cycle_tick = self.t

                        info["took_magic_damage"] = False

                    else:
                        # Magic projectile (28 damage)
                        self.hp -= 28
                        info["took_magic_damage"] = True

                        # Ensure we don't accidentally slam next tick
                        self.pending_slam = False
                        self.last_melee_cycle_tick = None

                        info["took_rise_damage"] = False
                        info["took_slam_damage"] = False

                    # Count that we performed a standard attack on this cycle tick
                    self.standard_attack_count += 1
                    
        # Handle gaze countdown and resolution
        if self.gaze_active:
            self.gaze_timer -= 1
            if self.gaze_timer <= 0:
                # resolve gaze
                if not self._is_pillar_safe(self.agent_pos):
                    self.hp -= self.gaze_damage
                    info["took_gaze_damage"] = True
                else:
                    info["took_gaze_damage"] = False
                self.gaze_active = False
                self.gaze_timer = 0

        # --- 3) Reward / termination ---
        done = False

        # Start with survival reward
        reward = self.step_reward

        # Penalize damage taken this tick (encourages dodging)
        damage_this_tick = 0

        if info.get("took_magic_damage", False):
            damage_this_tick += 28
        if info.get("took_rise_damage", False):
            damage_this_tick += 3
        if info.get("took_slam_damage", False):
            damage_this_tick += self.slam_damage
        if info.get("took_gaze_damage", False):
            damage_this_tick += self.gaze_damage

        # Scale factor: tune this. 0.1 means 28 dmg => -2.8 reward.
        reward -= 0.1 * damage_this_tick

        if self.hp <= 0:
            done = True
            reward = self.death_penalty

        # --- Finalize info with POST-resolution values (so playback is consistent) ---
        info["hp"] = self.hp
        info["gaze_active"] = self.gaze_active
        info["gaze_timer"] = self.gaze_timer

        self.t += 1
        if self.t >= self.max_steps:
            done = True  # survived the whole episode

        next_state = self.pos_to_state(self.agent_pos)
        return next_state, reward, done, info

    def render(self):
        """
        Console render:
        - # walls
        - B boss tiles
        - . open
        - m melee-range tiles
        - v vent tiles
        - P pillar-safe tiles
        - A agent
        """
        display = np.full(self.grid.shape, ".", dtype="<U2")

        display[self.grid == self.TILE_WALL] = "#"
        display[self.grid == self.TILE_BOSS] = "B"
        display[self.grid == self.TILE_MELEE] = "m"
        display[self.grid == self.TILE_VENT] = "v"
        display[self.grid == self.TILE_PILLAR] = "P"

        # Place agent on top
        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        display[ar, ac] = "A"

        # Print the grid
        print("\n".join(" ".join(row) for row in display))
        print(f"tick={self.t} hp={self.hp} gaze_active={self.gaze_active} gaze_timer={self.gaze_timer}")


if __name__ == "__main__":
    import time

    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env = DukeSurvivalEnv(max_steps=200, seed=0)
    state = env.reset()

    delay_seconds = 0.10
    steps_to_show = 80

    for i in range(steps_to_show):
        # Print a frame separator so it's easy to see each tick
        print("\n" + "=" * 60)
        print(f"FRAME {i}")
        

        # Render the arena (this MUST print the grid)
        env.render()
        print(f"phase: telegraph={env.t % 5 == 0}  slam={env.t % 5 == 1}")

        # Choose an action (random demo)
        action = int(env.rng.integers(0, env.n_actions))

        next_state, reward, done, info = env.step(action)

        print(f"action={action} ({ACTION_NAMES.get(action, action)})  reward={reward}  done={done}")
        print(f"info={info}")

        time.sleep(delay_seconds)

        state = next_state
        
        
        if done:
            print("\nEpisode ended.\n")
            break
