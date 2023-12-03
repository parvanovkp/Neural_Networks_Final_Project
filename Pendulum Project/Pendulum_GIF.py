import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio
import os

# Constants
b = 0.5 # damping coefficient
g = 9.81 # acceleration due to gravity
L = 6 # length of the pendulum
y0 = [np.pi/4, 0] # Initial conditions

# Damped pendulum differential equation
def damped_pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Time span for the solution
t_span = [0, 10]  # From t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 300)  # More points for smoother animation
sol = solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval)

# Function to get pendulum coordinates
def pendulum_coordinates(theta, L):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y

# Function to adjust line endpoint
def adjust_line_endpoint(x, y, L, bob_size):
    # Calculate the angle of the pendulum
    angle = np.arctan2(y, x)

    # Calculate the offset caused by the bob size
    offset = bob_size / 2200 * L  # Adjust the factor to match the bob size
    x_adjusted = x - offset * np.cos(angle)
    y_adjusted = y - offset * np.sin(angle)

    return x_adjusted, y_adjusted

# Generate and save frames
bob_size = 100
filenames = []
for i in range(len(sol.t)):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    # Plot solution curve
    axs[0].plot(sol.t, sol.y[0], color='lightblue')
    axs[0].scatter(sol.t[i], sol.y[0][i], color='green')  # Moving dot
    axs[0].set_xlim(0, 10)
    axs[0].set_ylim(-1.5, 1.5)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Theta')
    axs[0].set_title('Solution Curve')

    # Plot pendulum
    x, y = pendulum_coordinates(sol.y[0][i], L)
    x_adjusted, y_adjusted = adjust_line_endpoint(x, y, L, bob_size)
    axs[1].plot([0, x_adjusted], [0, y_adjusted], color='black')  # Adjusted Pendulum line
    axs[1].scatter(x, y, color='green', edgecolor='black', s=bob_size)  # Adjusted Pendulum bob
    axs[1].plot([-1*L, 1*L], [0, 0], color='grey', linewidth=3)  # Ceiling
    axs[1].set_xlim(-1.5*L, 1.5*L)
    axs[1].set_ylim(-1.5*L, L)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_title('Pendulum Motion')

    # Save frame
    filename = f'frame_{i}.png'
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# Create a GIF
with imageio.get_writer('pendulum_motion.gif', mode='I', fps=20) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        os.remove(filename)  # Removes file after append to save space