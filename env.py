import numpy as np
# import matplotlib.pyplot as plt

# Action mapping: int -> (row_change, col_change)
ACTION_DELTAS = {
    0: np.array([-1, 0]),  # Up
    1: np.array([0, 1]),   # Right
    2: np.array([1, 0]),   # Down
    3: np.array([0, -1])   # Left
    
}

class MazeEnv:
    def __init__(self):

        """
        Simple 4x4 Maze.
        
        0 = free cell
        1 = wall
        
        S = start, G = goal (just for mental picture)
        
        [S][ ][ ][#]
        [#][#][ ][#]
        [ ][ ][ ][ ]
        [ ][#][#][G]
        """
        
        self.grid = np.array([
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        
        self.start_pos = np.array([0, 0]) # Starting position (row, col)
        self.goal_pos = np.array([3, 3])  # Goal position (row, col)
        
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTION_DELTAS)
        
        # will be set in reset
        self.agent_pos = None
        print(self.grid)

    # --- helpers for state <-> index ---
    
    def pos_to_state(self, pos: np.ndarray) -> int:
        """Convert (row, col) position to integer state index."""
        r, c = int(pos[0]), int(pos[1])
        return r * self.n_cols + c
    
    def state_to_pos(self, state: int) -> np.ndarray:
        """Convert integer state index to (row, col) position."""
        r = state // self.n_cols
        c = state % self.n_cols
        return np.array([r, c])
    
    # --- standard RL-style API ---
    
    def reset(self) -> int:
        """Reset the environment to the starting state."""
        self.agent_pos = self.start_pos.copy()
        return self.pos_to_state(self.agent_pos)
    
    def step(self, action: int):
        """
        Take an action in the environment.
        
        Returns: next_state (int), reward (float), done (bool), info (dict)
        """
        
        assert action in ACTION_DELTAS, "Invalid action: {action}"
        
        #propose a new position
        new_pos = self.agent_pos + ACTION_DELTAS[action]
        
        # 1. check bounds
        r, c = int(new_pos[0]), int(new_pos[1])
        out_of_bounds = (
            r < 0 or r >= self.n_rows or
            c < 0 or c >= self.n_cols
        )
        
        if out_of_bounds or self.grid[r, c] == 1:
            # invalid move, stay in place & small penalty
            new_pos = self.agent_pos
            reward = -1.0
            done = False
        else:
            # valid move
            next_pos = new_pos
            
            if np.array_equal(next_pos, self.goal_pos):
                reward = 10.0
                done = True
            else:
                reward = -0.1 # negative reward to encourage shortest path
                done = False
                
        self.agent_pos = new_pos
        next_state = self.pos_to_state(self.agent_pos)
        info = {} # placeholder for later debugging
        return next_state, reward, done, info
    
    def render(self):
        """Print the maze with the agent's current position in the console."""
        display = np.array(self.grid, dtype=str)
        
        display[display == '0'] = " "  # free cell
        display[display == '1'] = "#"  # wall
        
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        ar, ac = self.agent_pos

        
        display[sr, sc] = 'S'  # Start
        display[gr, gc] = 'G'  # Goal

        # Marks the agent's current position (currently overrides S or G if on those cells)
        display[ar, ac] = 'A'  # Agent
        
        print("\n".join("".join(row) for row in display))
        print()

# --- simple test run ---

if __name__ == "__main__":
    env = MazeEnv()
    state = env.reset()
    env.render()

    # Try moving right 3 times then down 3 times
    actions = [1, 1, 1, 2, 2, 2]  # right, right, right, down, down, down

    for t, action in enumerate(actions):
        next_state, reward, done, _ = env.step(action)
        print(f"Step {t}: action={action}, state={next_state}, reward={reward}, done={done}")
        env.render()
        if done:
            print("Reached the goal!")
            break
