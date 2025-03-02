import numpy as np
from .GradientDescentOptimizer import GradientDescentOptimizer

class LinReg:
  def __init__(self):
    self.theta = None
    
  def fit(self, X, y, learning_rate=0.01, n_iterations=1000):
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    self.theta = np.random.randn(X_b.shape[1], 1)
    
    optimizer = GradientDescentOptimizer(learning_rate, n_iterations)
    self.theta = optimizer.optimize(X_b, y, self.theta)
    
  def predict(self, X):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(self.theta)