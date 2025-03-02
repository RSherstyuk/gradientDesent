import numpy as np

class GradientDescentOptimizer:
  def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0.1, regularization=None):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.lambda_ = lambda_
    self.regularization = regularization

  def optimize(self, X, y, theta):
    m = len(y)

    for iteration in range(self.n_iterations):
      gradients = 2 / m * X.T.dot(X.dot(theta) - y)

      if self.regularization == 'l2':
        gradients[1:] += (self.lambda_ / m) * theta[1:]
      elif self.regularization == 'l1':
        gradients[1:] += (self.lambda_ / m) * np.sign(theta[1:])

        theta = theta - self.learning_rate * gradients

        if iteration % 100 == 0:
          loss = self.compute_loss(X, y, theta)
          print(f"Iteration {iteration}: Loss = {loss}")

    return theta

  def compute_loss(self, X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1 / m) * np.sum((predictions - y) ** 2)

    if self.regularization == 'l2':
      loss += (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    elif self.regularization == 'l1':
      loss += (self.lambda_ / m) * np.sum(np.abs(theta[1:]))

    return loss