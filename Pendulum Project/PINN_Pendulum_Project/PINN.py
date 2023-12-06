import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.integrate import solve_ivp
import imageio.v2 as imageio
import os

# Create necessary directories
os.makedirs('training_plots/', exist_ok=True)
folder_path = '/Users/kaloyanparvanov/Downloads/Pendulum Project/PINN_Pendulum_Project/training_plots/'

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
b = 0.5 # damping coefficient
g = 9.81 # acceleration due to gravity
L = 6 # length of the pendulum
y0 = [np.pi/4, 0] # Initial conditions

# Differential Equation
def damped_pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Generate and solve ODE
def generate_ode_solution(t_span, y0, t_eval):
    return solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval)

# Neural Network Model
class DampedPendulumModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_list = [tf.keras.layers.Dense(units=32, activation='tanh') for _ in range(3)]
        self.output_layer = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)

# Physics-Informed Loss Function
def compute_loss(model, t_data, y0):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_data)
        predictions = model(t_data)
        theta, omega = predictions[:, 0:1], predictions[:, 1:2]

    # Compute gradients outside the context of the tape
    dtheta_dt = tape.gradient(theta, t_data, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    domega_dt = tape.gradient(omega, t_data, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    del tape  # Delete the persistent tape after gradients are computed

    # Damped pendulum equation
    damped_eq = domega_dt + b * omega + (g / L) * tf.sin(theta)

    # Loss for ODE and initial conditions
    ode_loss = tf.reduce_mean(tf.square(dtheta_dt - omega) + tf.square(damped_eq))
    ic_loss = tf.reduce_mean(tf.square(predictions[0] - y0))

    return ode_loss + ic_loss

# Training Function
def train(model, optimizer, t_data, y0, epochs, t_eval, theta_data, omega_data):
    loss_history = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, t_data, y0)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_history.append(loss.numpy())
        if epoch % 1000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
            plot_predictions(model, t_eval, epoch, folder_path, theta_data, omega_data)
    return loss_history

# Plotting Functions
def plot_predictions(model, t_eval, epoch, folder_path, theta_data, omega_data):

    # Predicted solution (uniformly sampled from [0,10])
    predicted_solution = model(t_eval).numpy()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=400)

    # First subplot for theta
    axs[0].plot(t_eval, predicted_solution[:, 0], 'orange', linestyle='--', label='Predicted Theta')
    axs[0].plot(t_eval, theta_data, color='lightblue', label='True Theta')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Theta')
    axs[0].set_ylim(-0.45, 0.85)
    axs[0].set_xlim(0, 10)
    # Handle 'final' case for title
    title_suffix = 'final' if epoch == 'final' else f'Epoch {epoch+1}'
    axs[0].set_title(f'Theta: Predicted vs Actual {title_suffix}')
    axs[0].legend()

    # Second subplot for omega
    axs[1].plot(t_eval, predicted_solution[:, 1], 'orange', linestyle='--', label='Predicted Omega')
    axs[1].plot(t_eval, omega_data, color='lightblue', label='True Omega')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Omega')
    axs[1].set_ylim(-0.85, 0.45)
    axs[1].set_xlim(0, 10)
    axs[1].set_title(f'Omega: Predicted vs Actual {title_suffix}')
    axs[1].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(f'{folder_path}prediction_{epoch+1}.png')
    plt.close(fig)

def plot_log_error(true_values, predicted_values, t_eval_predict, title, filename):
    error = np.log10(np.abs(true_values - predicted_values) + 1e-10)  # Log error
    plt.figure(figsize=(10, 8), dpi=400)
    plt.plot(t_eval_predict, error)
    plt.xlabel('Time')
    plt.ylabel('Log Base 10 Error')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_interval_predictions(model, t_span_predict, y0):
    # Generate true solution and predictions
    t_eval_predict = np.linspace(t_span_predict[0], t_span_predict[1], 1000).reshape(-1, 1)
    sol_predict = generate_ode_solution(t_span_predict, y0, t_eval_predict.flatten())
    model_predictions = model(t_eval_predict).numpy()

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=400)

    # Theta plot
    axs[0].plot(t_eval_predict, sol_predict.y[0], color='lightblue', label='True Theta')
    axs[0].plot(t_eval_predict, model_predictions[:, 0], 'orange',
                linestyle='--', label='Predicted Theta')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Theta')
    axs[0].set_title('Theta Predicted vs Actual Outside Training')
    axs[0].legend()

    # Omega plot
    axs[1].plot(t_eval_predict, sol_predict.y[1], color='lightblue', label='True Omega')
    axs[1].plot(t_eval_predict, model_predictions[:, 1], 'orange',
                linestyle='--', label='Predicted Omega')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Omega')
    axs[1].set_title('Omega Predicted vs Actual Outside Training')
    axs[1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('pinn_prediction_outside_training.png')
    plt.close(fig)

# Create GIF from Training Plots
def create_gif(image_folder, gif_name):
    filenames = [f'{image_folder}prediction_{i+1}.png' for i in range(0, epochs, 1000)]
    with imageio.get_writer(gif_name, mode='I', duration=1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            #os.remove(filename)

# Data Preparation
t_eval_train = tf.constant(np.linspace(0, 20, 500).reshape(-1, 1), dtype=tf.float32)
t_span_plot, t_eval_plot = [0, 10], np.linspace(0, 10, 1000).reshape(-1, 1) #for plotting purposes
sol = generate_ode_solution(t_span_plot, y0, t_eval_plot.flatten()) #for plotting purposes
theta_data_plot, omega_data_plot = sol.y[0], sol.y[1] #for plotting purposes

# Model Training
model = DampedPendulumModel()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
epochs = 30001
loss_history = train(model, optimizer, t_eval_train, y0, epochs, t_eval_plot, theta_data_plot, omega_data_plot)

# Plot Loss History
plt.figure(figsize=(10, 8), dpi=400)
plt.plot(np.log10(loss_history))
plt.xlabel('Epoch')
plt.ylabel('Log Base 10 Loss')
plt.title('PINN Model Log-Loss History')
plt.savefig('model_loss_history.png')
plt.show()

# Create GIF
create_gif(folder_path, 'training_progress.gif')

# Predictions for [0, 20]
t_span_predict = [0, 20]
t_eval_predict = np.linspace(t_span_predict[0], t_span_predict[1], 1000).reshape(-1, 1)
model_predictions = model(t_eval_predict).numpy()

# True solution for [0, 20]
sol_predict = generate_ode_solution(t_span_predict, y0, t_eval_predict.flatten())
numerical_theta = sol_predict.y[0]
numerical_omega = sol_predict.y[1]

# Plot Predicted vs True and Log Error
plot_log_error(numerical_theta, model_predictions[:, 0], t_eval_predict,
               'Log Error of Theta Over Time','log_error_theta.png')
plot_log_error(numerical_omega, model_predictions[:, 1], t_eval_predict,
               'Log Error of Omega Over Time', 'log_error_omega.png')

# True solution vs predicted for [0,20]
plot_interval_predictions(model, t_span_predict, y0)