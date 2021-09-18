import numpy as np

np.random.seed(0)

X = np.array([[1.0, 0.7]])
y_true = np.array([1.80])

def initialize_parameters(n_x,n_h,n_y): 
  W1 = np.random.randn(n_h, n_x) 
  W2 = np.random.randn(n_h ,n_y) 
  return W1, W2

def forward_propagation(X , W1, W2):
  H = np.dot(X, W1) 
  y_pred = np.dot(H, W2)
  return H ,y_pred

def calculate_error(y_true, y_pred): 
  return y_pred-y_true

def backpropagation(X , W1 ,W2, learning_rate = 0.01, iters = 1000, precision = 0.0000001):
  H , y_pred = forward_propagation(X , W1, W2) 

  for i in range(iters): 
    error = calculate_error(y_true, y_pred)
    W2 = W2 - learning_rate * error * H.T 
    W1 = W1 - learning_rate * error * X.T * W2.T
    _, y_pred = forward_propagation(X , W1 , W2)
    print("Iter {}, y_pred: {}, error: {}".format(i , y_pred[0][0],calculate_error(y_true, y_pred)[0]))

    if abs(error) < precision:
      break

  return W1, W2

def predict(X, W1, W2):
  _, y_pred = forward_propagation(X , W1 , W2)
  return y_pred

def build_model():
  W1, W2 = initialize_parameters(2, 2, 1)
  W1,W2 = backpropagation(X, W1, W2)
  model = {'W1' : W1, 'W2' : W2}
  return model

build_model() 
