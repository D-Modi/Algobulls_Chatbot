import numpy as np
import cvxpy as cp

# Example data (expected returns and covariance matrix)
expected_returns = np.array([0.10, 0.08, 0.12])  # Mean returns
covariance_matrix = np.array([[0.10, 0.05, 0.03],
                               [0.05, 0.12, 0.07],
                               [0.03, 0.07, 0.15]])  # Covariance matrix

# Define variables
weights = cp.Variable(len(expected_returns))

# Define objective function (Minimize portfolio risk for a given level of return)
portfolio_return = expected_returns @ weights
portfolio_risk = cp.quad_form(weights, covariance_matrix)
objective = cp.Minimize(portfolio_risk)

# Define constraints (sum of weights equals 1)
constraints = [cp.sum(weights) == 1]

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Optimal portfolio weights
optimal_weights = weights.value
print("Optimal Portfolio Weights:", optimal_weights)

# Portfolio expected return and volatility
expected_return = portfolio_return.value
volatility = np.sqrt(portfolio_risk.value)
print("Expected Portfolio Return:", expected_return)
print("Portfolio Volatility:", volatility)
