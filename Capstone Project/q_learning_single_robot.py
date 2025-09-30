# q_learning_dynamic_final.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ---------------- Parameters ----------------
GRID_SIZE = 10
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Static and dynamic obstacles
static_obstacles = [(2, 3), (5, 5), (3, 7)]
dynamic_obstacles = [(4, 4), (7, 2)]
directions = [(0, 1), (1, 0)]  # directions for dynamic obstacles

for (x, y) in static_obstacles:
    grid[x, y] = 1

start = (0, 0)
goal = (9, 9)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

# ---------------- Helper Functions ----------------
def in_bounds(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def update_dynamic_obstacles():
    """Move dynamic obstacles and bounce at edges."""
    global dynamic_obstacles, directions
    new_positions = []
    new_directions = []
    for (pos, d) in zip(dynamic_obstacles, directions):
        x, y = pos
        dx, dy = d
        nx, ny = x + dx, y + dy
        if not in_bounds((nx, ny)) or grid[nx, ny] == 1:
            dx, dy = -dx, -dy  # bounce
            nx, ny = x + dx, y + dy
        new_positions.append((nx, ny))
        new_directions.append((dx, dy))
    dynamic_obstacles[:] = new_positions
    directions[:] = new_directions

def step(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    next_pos = (nx, ny)
    if not in_bounds(next_pos) or grid[nx, ny] == 1:
        return state, -10, False
    if next_pos in dynamic_obstacles:
        return state, -15, False
    if next_pos == goal:
        return next_pos, 20, True
    return next_pos, -1, False

# ---------------- Q-learning ----------------
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)), dtype=float)
alpha, gamma, epsilon = 0.1, 0.9, 0.3
min_epsilon, decay = 0.05, 0.995
episodes, max_steps = 800, 150

for ep in range(episodes):
    state = start
    for _ in range(max_steps):
        x, y = state
        if random.random() < epsilon:
            action = random.randint(0, len(ACTIONS)-1)
        else:
            action = int(np.argmax(q_table[x, y]))
        update_dynamic_obstacles()
        next_state, reward, done = step(state, action)
        nx, ny = next_state
        old = q_table[x, y, action]
        future = 0 if done else np.max(q_table[nx, ny])
        q_table[x, y, action] = old + alpha*(reward + gamma*future - old)
        state = next_state
        if done:
            break
    epsilon = max(min_epsilon, epsilon*decay)

# ---------------- Test Policy ----------------
state = start
path = [state]
obs_history = [list(dynamic_obstacles)]
for _ in range(200):
    update_dynamic_obstacles()
    x, y = state
    action = int(np.argmax(q_table[x, y]))
    next_state, reward, done = step(state, action)
    if next_state == state and reward < 0:
        break
    path.append(next_state)
    obs_history.append(list(dynamic_obstacles))
    state = next_state
    if done:
        break

print("Dynamic test path length:", len(path), "Reached goal:", path[-1]==goal)

# ---------------- Animation Function ----------------
def animate_dynamic(grid, path, static_obstacles, obs_history, goal, interval=300):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE,1))
    ax.set_yticks(np.arange(-0.5, GRID_SIZE,1))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.7)
    ax.invert_yaxis()

    for (x, y) in static_obstacles:
        ax.scatter(y, x, s=220, marker="s", color="black")
    sx, sy = path[0]
    gx, gy = goal
    ax.scatter(sy, sx, s=250, marker="s", color="green")
    ax.scatter(gy, gx, s=300, marker="*", color="red")

    dyn_scat = ax.scatter([], [], s=220, marker="s", color="orange")
    robot_dot, = ax.plot([], [], "bo", markersize=12)
    path_line, = ax.plot([], [], "b-", linewidth=2, alpha=0.5)
    title = ax.set_title("Robot Path Planning (Dynamic Environment)", fontsize=14)

    def init():
        robot_dot.set_data([], [])
        path_line.set_data([], [])
        dyn_scat.set_offsets(np.empty((0,2)))
        return robot_dot, path_line, dyn_scat

    def update(frame):
        xs = [p[0] for p in path[:frame+1]]
        ys = [p[1] for p in path[:frame+1]]
        robot_dot.set_data([ys[-1]], [xs[-1]])
        path_line.set_data(ys, xs)
        obs = obs_history[frame]
        obs_xy = np.array([[y, x] for (x,y) in obs])
        dyn_scat.set_offsets(obs_xy)
        title.set_text(f"Step {frame+1}/{len(path)}")
        return robot_dot, path_line, dyn_scat, title

    ani = FuncAnimation(fig, update, frames=len(path),
                        init_func=init, blit=False,
                        interval=interval, repeat=False)
    plt.show(block=True)
    return ani

# ---------------- Save Animation Function ----------------
def save_animation(ani, mp4_name="robot_dynamic.mp4", gif_name="robot_dynamic.gif", fps=5):
    # Save MP4
    ani.save(mp4_name, writer="ffmpeg", fps=fps)
    print(f"Saved MP4: {mp4_name}")
    # Save GIF
    ani.save(gif_name, writer="pillow", fps=fps)
    print(f"Saved GIF: {gif_name}")

# ---------------- Run Animation & Save ----------------
ani = animate_dynamic(grid, path, static_obstacles, obs_history, goal)
save_animation(ani)
