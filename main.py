import numpy as np
from src.LinReg import LinReg

if __name__ == "__main__":
  np.random.seed(42)
  X = 2 * np.random.rand(100, 1)  # 100 точек в диапазоне [0, 2]
  y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + шум

  model = LinReg()
  model.fit(X, y, learning_rate=0.1, n_iterations=1000)


  X_new = np.array([[0], [2]])  # Новые данные для предсказания
  predictions = model.predict(X_new)

  # Вывод результатов
  print("Оптимальные параметры модели:", model.theta.ravel())
  print("Предсказания для X_new:", predictions.ravel())