import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jax import jit
import numpy as np
############################# Parameters ###################################
dt = 0.001
g = 9.81
restitution = 0.8  # k
friction = 0.0

# Box
box_size = 1  # half-width/height of the box

# boundaries
wall_left = 0
wall_right = 10
wall_bottom = 0
wall_top = 10

# Initial conditions
initial_pos = jnp.array([1.0, 1.0])  # x, y pos
initial_vel = jnp.array([2.0, 2.0])  # Initial vel

################################################################################################
@jit
def physics_derivatives(state, t):
    """
    state = [x, y, vx, vy]
    Returns [vx, vy, ax, ay]
    """
    pos = state[:2]
    vel = state[2:]
    
    #gravity and air resistance
    accel = jnp.array([
        -friction * vel[0],  #Air resistance x
        -g - friction * vel[1]  #Gravity + air resistance y
    ])
    
    return jnp.concatenate([vel, accel])
################################################################################################
@jit
def rk4_step(state, t, dt):
    """
    Runge-Kutta 4th order integration step
    """
    k1 = physics_derivatives(state, t)
    k2 = physics_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = physics_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = physics_derivatives(state + dt * k3, t + dt)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
################################################################################################
@jit
def handle_collisions(state):
    """
    Handle collisions with walls
    """
    pos = state[:2]
    vel = state[2:]
    
    new_pos = pos
    new_vel = vel
    
    new_pos = new_pos.at[0].set(
        jnp.where(pos[0] - box_size < wall_left,
                  wall_left + box_size, 
                  new_pos[0]))
    
    new_vel = new_vel.at[0].set(
        jnp.where(pos[0] - box_size < wall_left,
                  -restitution * vel[0],
                  new_vel[0]))
    
    # Right wall
    new_pos = new_pos.at[0].set(
        jnp.where(pos[0] + box_size > wall_right,
                  wall_right - box_size,
                  new_pos[0])
    )
    new_vel = new_vel.at[0].set(
        jnp.where(pos[0] + box_size > wall_right,
                  -restitution * vel[0],
                  new_vel[0])
    )
    
    # Bottom wall collision
    new_pos = new_pos.at[1].set(
        jnp.where(pos[1] - box_size < wall_bottom,
                  wall_bottom + box_size,
                  new_pos[1])
    )
    new_vel = new_vel.at[1].set(
        jnp.where(pos[1] - box_size < wall_bottom,
                  -restitution * vel[1],
                  new_vel[1])
    )
    
    # Top wall collision
    new_pos = new_pos.at[1].set(
        jnp.where(pos[1] + box_size > wall_top,
                  wall_top - box_size,
                  new_pos[1])
    )
    new_vel = new_vel.at[1].set(
        jnp.where(pos[1] + box_size > wall_top,
                  -restitution * vel[1],
                  new_vel[1])
    )
    
    return jnp.concatenate([new_pos, new_vel])################################################################
@jit
def simulate_step(state, t):
    """
    Single simulation step: integrate then handle collisions
    """
    # RK4 integration
    new_state = rk4_step(state, t, dt)
    
    # Handle collisions
    new_state = handle_collisions(new_state)
    
    return new_state
################################################################################################
def run_simulation(total_time = 10.0):  ######################## IMPORTANT

    num_steps = int(total_time / dt)
    
    # Initialize state [x, y, vx, vy]
    state = jnp.concatenate([initial_pos, initial_vel])
    
################################# Storage for animation ################################
    positions = []
    times = []
################################################################################################
    
    t = 0.0 ## INIT TIME 0.0
    for i in range(num_steps):
## Store current position for animation 
        if i % 50 == 0:  # Store every 50th frame for smoother animation
            positions.append(state[:2].copy())
            times.append(t)
        
        # Simulate one step
        state = simulate_step(state, t)
        t += dt
        
        # Stop if box is at rest at bottom
        if (abs(state[3]) < 0.1 and abs(state[2]) < 0.1 and 
            state[1] <= wall_bottom + box_size + 0.01):
            break
    
    return np.array(positions), np.array(times)
################################################################################################
# Run simulation
positions, times = run_simulation()

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(wall_left - 1, wall_right + 1)
ax.set_ylim(wall_bottom - 1, wall_top + 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.5)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Box Physics Simulation with RK4 Integration')

# Draw walls
wall_props = dict(linewidth=3, color='black')
ax.axvline(x=wall_left, **wall_props)
ax.axvline(x=wall_right, **wall_props)
ax.axhline(y=wall_bottom, **wall_props)
ax.axhline(y=wall_top, **wall_props)

# Box visualization
box_patch = plt.Rectangle((0, 0), 2*box_size, 2*box_size, alpha=0.7)
ax.add_patch(box_patch)

# Trail line
trail_line, = ax.plot([], [], 'b-', alpha = 0.5, linewidth=1, label='Trail')
position_dot, = ax.plot([], [], 'ro', markersize=8, label='Center')

# Text displays
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
info_text = ax.text(0.02, 0.88, 'RK4 Integration\nGravity + Collisions', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.legend(loc='upper right')
################################################################################################################################

def animate(frame):
    if frame >= len(positions):
        return box_patch, trail_line, position_dot, time_text
    
    pos = positions[frame]
    time = times[frame]
    
    # Update box position
    box_patch.set_xy([pos[0] - box_size, pos[1] - box_size])
    
    # Update trail
    if frame > 0:
        trail_x = positions[:frame+1, 0]
        trail_y = positions[:frame+1, 1]
        trail_line.set_data(trail_x, trail_y)
    
    # Update center dot
    position_dot.set_data([pos[0]], [pos[1]])
    
    # Update time display
    time_text.set_text(f'Time: {time:.2f}s\nFrame: {frame}\nPos: ({pos[0]:.2f}, {pos[1]:.2f})')
    
    return box_patch, trail_line, position_dot, time_text
################################################################################################
print(f"Creating animation with {len(positions)} frames...")
anim = animation.FuncAnimation(fig, animate, frames=len(positions), 
                              interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()