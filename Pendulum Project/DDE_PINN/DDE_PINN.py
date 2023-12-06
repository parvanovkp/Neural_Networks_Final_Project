import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.integrate import solve_ivp
import imageio.v2 as iio
import tensorflow as tf
import os


# Constants for the damped pendulum
b = 0.5  # damping coefficient
g = 9.81 # acceleration due to gravity
L = 6    # length of the pendulum
y0 = [np.pi / 4, 0]  # Initial conditions

# Damped pendulum differential equation for solve_ivp
def damped_pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Time span for the solution
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Solve the ODE for the reference solution
sol = solve_ivp(damped_pendulum, t_span, y0, t_eval=t_eval)
theta_data_ref = sol.y[0]
omega_data_ref = sol.y[1]

# Define the damped pendulum system for DeepXDE
def damped_pendulum_system(t, y):
    theta, omega = y[:, 0:1], y[:, 1:2]
    dtheta_dt = dde.grad.jacobian(y, t, i=0)
    domega_dt = dde.grad.jacobian(y, t, i=1)
    return [dtheta_dt - omega,
            domega_dt - (-b * omega - (g / L) * tf.sin(theta))]

def boundary(t, on_initial):
    return on_initial and np.isclose(t[0], 0)

# Geometry of the problem (time domain)
geom = dde.geometry.TimeDomain(0, 20)

# Initial conditions
ic1 = dde.IC(geom, lambda X: tf.constant(y0[0]), boundary, component=0)  # theta(0) = pi/4
ic2 = dde.IC(geom, lambda X: tf.constant(y0[1]), boundary, component=1)  # omega(0) = 0

# Data for the PDE
data = dde.data.PDE(geom, damped_pendulum_system, [ic1, ic2], num_domain=400, num_boundary=2, num_test=100)

# Neural network configuration
layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"

# Define the neural network
net = dde.maps.PFNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

class SavePredictionCallback(dde.callbacks.Callback):
    def __init__(self, t_data, true_solution, freq=1000):
        super(SavePredictionCallback, self).__init__()
        self.t_data = t_data
        self.true_solution = true_solution
        self.freq = freq

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # Save at the first epoch and then every `freq` epochs
        if (epoch-1) % self.freq == 0:
            print(f"Epoch {epoch}: Saving plot...")
            y_pred = self.model.predict(self.t_data)
            self.save_plot(y_pred, epoch)

    def save_plot(self, y_pred, epoch):
        plt.figure(figsize=(10, 8))
        plt.plot(self.t_data, y_pred[:, 0], '--', label='Predicted Theta', color="orange")
        plt.plot(self.t_data, self.true_solution[:, 0], label='True Theta', color="lightblue")
        plt.xlabel('Time')
        plt.ylabel('Theta')
        plt.title(f'Epoch {epoch}')
        plt.legend()
        filename = f'theta_epoch_{epoch}.png'
        plt.savefig(filename)
        plt.close()
        print(f"File saved: {filename}")


# Prepare true solution and time data for the callback
t_data = np.linspace(0, 20, 2000).reshape(-1, 1)
true_solution = np.stack([theta_data_ref, omega_data_ref], axis=1)

# Instantiate the callback
callback = SavePredictionCallback(t_data, true_solution)


# Train the model with the callback
epochs = 150001
losshistory, train_state = model.train(epochs=epochs, callbacks=[callback])

# Compile saved plots into a GIF after training
images = []
filenames = [f'theta_epoch_{i+1}.png' for i in range(0, epochs, 1000)] + [f'theta_epoch_{epochs}.png']
with iio.get_writer('training_progress_DDE_PINN.gif', mode='I', duration=1, loop=0) as writer:
    for filename in filenames:
        image = iio.imread(filename)
        writer.append_data(image)
        os.remove(filename)  # Optionally remove the file after adding it to the GIF


# New time span for prediction
t_span_predict = [0, 10]
t_eval_predict = np.random.uniform(t_span_predict[0], t_span_predict[1], 1000)
t_eval_predict = np.sort(t_eval_predict)
t_eval_predict = t_eval_predict.reshape(-1, 1)


# Use the model to make predictions
# Convert TensorFlow tensor to NumPy array for prediction
model_predictions = model.predict(t_eval_predict)
predicted_theta = model_predictions[:, 0]
predicted_omega = model_predictions[:, 1]


# Numerical solution for the new time span
sol_predict = solve_ivp(damped_pendulum, t_span_predict, y0, t_eval=t_eval_predict.flatten())
numerical_theta = sol_predict.y[0]
numerical_omega = sol_predict.y[1]


# Plot for theta
plt.figure(figsize=(12, 6))
plt.plot(t_eval_predict, predicted_theta, label='Predicted Theta', linestyle='--')
plt.plot(t_eval_predict, numerical_theta, label='Numerical Theta', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.title('Comparison of Predicted and Numerical Theta')
plt.legend()
plt.show()

# Plot for omega
plt.figure(figsize=(12, 6))
plt.plot(t_eval_predict, predicted_omega, label='Predicted Omega', linestyle='--')
plt.plot(t_eval_predict, numerical_omega, label='Numerical Omega', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Omega')
plt.title('Comparison of Predicted and Numerical Omega')
plt.legend()
plt.show()