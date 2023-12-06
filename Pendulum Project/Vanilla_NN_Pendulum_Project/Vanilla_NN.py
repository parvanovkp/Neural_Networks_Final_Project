import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.integrate import solve_ivp
import imageio.v2 as iio
import os


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Vanilla Loss Function
def compute_vanilla_loss(predictions, actual_theta, actual_omega):
    return tf.reduce_mean(tf.square(predictions[:, 0] - actual_theta) + tf.square(predictions[:, 1] - actual_omega))

# Vanilla NN Training Function
def train_vanilla(model, optimizer, t_data, theta_data, omega_data, epochs):
    train_loss_record = []
    plot_filenames = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(tf.constant(t_data.reshape(-1, 1), dtype=tf.float32))
            loss = compute_vanilla_loss(predictions, theta_data, omega_data)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_record.append(loss.numpy())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
            plot_filename = f'plot_epoch_{epoch+1}.png'
            plot_predictions_vs_actual(t_data, predictions, theta_data, omega_data, plot_filename, epoch)
            plot_filenames.append(plot_filename)

    return train_loss_record, plot_filenames

def plot_predictions_vs_actual(t_data, predictions, theta_data, omega_data, filename, epoch):
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()

    # Set a figure size and resolution
    plt.figure(figsize=(16, 8), dpi=400)
    for i, (data, title, ylim) in enumerate(zip([theta_data, omega_data], ['Theta', 'Omega'], [(-0.45, 0.85), (-.85, 0.45)])):
        plt.subplot(1, 2, i+1)
        plt.plot(t_data, predictions[:, i], '--', label='Predicted', color='orange')
        plt.plot(t_data, data, label='Actual', color='lightblue')
        plt.xlabel('Time')
        plt.ylabel(title)
        plt.title(f'{title}: Predicted vs Actual {epoch + 1}')
        plt.ylim(ylim)  # Set specific y-axis limits for each subplot
        plt.xlim(0, 10)
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_predictions_outside_training(t_data, predictions, theta_data, omega_data, filename):
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()

    plt.figure(figsize=(16, 8), dpi=400)
    titles = ['Theta', 'Omega']
    y_lims = [(-0.45, 0.85), (-0.85, 0.45)]  # y-axis limits (for plotting purposes)

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.plot(t_data, predictions[:, i], '--', label='Predicted', color='orange')
        plt.plot(t_data, [theta_data, omega_data][i], label='Actual', color='lightblue')
        plt.xlabel('Time')
        plt.ylabel(titles[i])
        plt.title(f'{titles[i]}: Predicted vs Actual Outside Training')
        plt.ylim(y_lims[i])  # Set specific y-axis limits for each subplot
        plt.xlim(t_data[0], t_data[-1])
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to plot log error
def plot_log_error(t_data, predictions, actual_data, filename):
    error = np.log10(np.abs(predictions - actual_data) + 1e-10)  # adding a small constant to avoid log(0)
    plt.figure(figsize=(10, 8), dpi=400)
    plt.plot(t_data, error)
    plt.xlabel('Time')
    plt.ylabel('Log Base 10 Error')
    plt.title('Log Error Over Time')
    plt.savefig(filename)
    plt.close()
def create_gif(plot_filenames, gif_name):
    with iio.get_writer(gif_name, mode='I', duration=5, loop=0, fps=6) as writer:
        for filename in plot_filenames:
            image = iio.imread(filename)
            writer.append_data(image)
            #os.remove(filename)

# Function to generate data for a new time interval
def generate_new_data(time_interval):
    t_eval_new = np.linspace(time_interval[0], time_interval[1], 1000)
    sol_new = solve_ivp(damped_pendulum, time_interval, y0, t_eval=t_eval_new)
    return t_eval_new, sol_new.y[0], sol_new.y[1]

# Time span for the solution [Training Data]
t_span = [0, 10]  # From t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 250)  # Evaluate at 1000 time points

# Solve the ODE
sol = solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval)

# Extract the solution
t_data = sol.t
theta_data = sol.y[0]
omega_data = sol.y[1]

# Initialize model and optimizer
vanilla_model = DampedPendulumModel()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Train the vanilla model
epochs = 30001  # Number of epochs
vanilla_loss_history, plot_filenames = train_vanilla(vanilla_model, optimizer, t_data, theta_data, omega_data, epochs)

# Plot vanilla model loss history
plt.figure(figsize=(10, 8), dpi=400)
plt.plot(np.log10(vanilla_loss_history))
plt.xlabel('Epoch')
plt.ylabel('Log Base 10 Loss')
plt.title('Vanilla Model Log-Loss History')
plt.savefig('Vanilla_Model_Loss_History.png')
plt.show()

# Create GIF
create_gif(plot_filenames, 'training_progress_VNN.gif')

# Generate new data for interval [0, 20]
t_data_new, theta_data_new, omega_data_new = generate_new_data([0, 20])

# Predict with the trained model
predictions_new = vanilla_model(tf.constant(t_data_new.reshape(-1, 1), dtype=tf.float32))

# Use the new function for plotting outside training range
plot_predictions_outside_training(t_data_new, predictions_new,
                                  theta_data_new, omega_data_new, 'prediction_outside_training.png')

# Plot log error for theta
plot_log_error(t_data_new, predictions_new[:, 0], theta_data_new, 'log_error_theta.png')

# Plot log error for omega
plot_log_error(t_data_new, predictions_new[:, 1], omega_data_new, 'log_error_omega.png')

# Display the prediction outside training plot
plt.imshow(plt.imread('prediction_outside_training.png'))
plt.axis('off')
plt.show()

# Display the log error plots
plt.imshow(plt.imread('log_error_theta.png'))
plt.axis('off')
plt.show()

plt.imshow(plt.imread('log_error_omega.png'))
plt.axis('off')
plt.show()