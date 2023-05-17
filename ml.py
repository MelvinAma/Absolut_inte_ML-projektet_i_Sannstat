import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set the seed for reproducibility
np.random.seed(0)

# Define the true weights and noise level
w_true = np.array([0, 1.5, -0.8])

for sigma_squared in [0.2, 0.4, 0.6]:
    # Generate the input data
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X0 = np.ones_like(X1).ravel()  # Create a flattened array of ones
    X = np.stack([X0, X1.ravel(), X2.ravel()], axis=-1)

    # Generate the target data with noise
    t = np.einsum('ij,j->i', X, w_true) + np.random.normal(0, sigma_squared, size=X1.size)


    # Define the mask for selecting the test data
    mask = (np.abs(X1) > 0.3) & (np.abs(X2) > 0.3)
    mask = mask.ravel()  # Flatten the mask

    # Split the data
    X_train = X[~mask]
    t_train = t[~mask]
    X_test = X[mask]
    t_test = t[mask]


    # Create and fit the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, t_train)

    # Print variation number
    print()
    print("σ^2 =", sigma_squared)

    # Print the estimated weights
    print("Estimated weights:", model.coef_)

    # Make predictions on the test data
    t_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(t_test, t_pred)
    print("Mean squared error:", mse)

    # Define the prior mean and covariance
    prior_mean = np.zeros(3)
    prior_cov = np.eye(3) / sigma_squared

    # Calculate the posterior mean and covariance
    posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + X_train.T @ X_train / sigma_squared)
    posterior_mean = posterior_cov @ (np.linalg.inv(prior_cov) @ prior_mean + X_train.T @ t_train / sigma_squared)

    # Print the posterior mean (the estimated weights)
    print("Posterior mean:", posterior_mean)

    # Make predictions on the training data
    t_train_pred = X_train @ posterior_mean

    # Calculate the variance of the predictions
    var_train = np.var(t_train_pred)
    var_test = np.var(t_pred)

    # Print the variances
    print("Variance of training predictions:", var_train)
    print("Variance of test predictions:", var_test)

    # Reshape the target data to 2D for plotting
    t_2d = t.reshape(X1.shape)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the data points
    im = ax.imshow(t_2d, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')

    # Add a colorbar
    fig.colorbar(im, ax=ax)

    # Set the title
    ax.set_title(f"Data points for σ² = {sigma_squared}")

    # Show the plot
    plt.show()

