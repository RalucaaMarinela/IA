import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def train_perceptron(X,Y, epochs, lr):
    W = np.zeros(2)
    b = 0
    no_samps = X.shape[0]
    accuracy = 0.0
    for _ in range (epochs):
        X,Y = shuffle(X,Y)
        for i in range(no_samps):
            y_hat = np.dot(X[i][:],W)  + b
            less = (y_hat - Y[i]) **2
            W = W - lr * (y_hat - Y[i]) * X[i][:]
            b = b - lr * (y_hat - Y[i])
            accuracy = np.mean((np.sign(np.dot(X, W) + b) == Y))
    return W,b,accuracy

def fwd(x, W_1, W_2, b_1, b_2):
    z_1 = x * W_1 + b_1
    a_1  = np.tanh(z_1)
    z_2 = a_1 * W_2 + b_2
    a_2 = np.sigmoid(z_2)
    return z_1, a_1, z_2, a_2
def backward(a_1, a_2, z_1, W_2, X, Y, num_samples):
    dz_2 = a_2 - Y # derivata functiei de pierdere (logistic loss) in functie de z
    dw_2 = (a_1.T * dz_2) / num_samples # np.dot
    # der(L/w_2) = der(L/z_2) * der(dz_2/w_2) = dz_2 * der((a_1 * W_2 + b_2)/ W_2)
    db_2 = sum(dz_2) / num_samples  # np.sum
    # der(L/b_2) = der(L/z_2) * der(z_2/b_2) = dz_2 * der((a_1 * W_2 + b_2)/ b_2)
    # primul strat
    da_1 = dz_2 * W_2.T # np.dot
    # der(L/a_1) = der(L/z_2) * der(z_2/a_1) = dz_2 * der((a_1 * W_2 + b_2)/ a_1)
    dz_1 = da_1 * np.tanh(z_1)
    # der(L/z_1) = der(L/a_1) * der(a_1/z1) = da_1 .* der((tanh(z_1))/ z_1)
    dw_1 = X.T * dz_1 / num_samples
    # der(L/w_1) = der(L/z_1) * der(z_1/w_1) = dz_1 * der((X * W_1 + b_1)/ W_1)
    db_1 = sum(dz_1) / num_samples
    # der(L/b_1) = der(L/z_1) * der(z_1/b_1) = dz_1 * der((X * W_1 + b_1)/ b_1)
    return dw_1, db_1, dw_2, db_2
def plot_decision(X_, W_1, W_2, b_1, b_2):
     # sterge continutul ferestrei
     plt.clf()
     # ploteaza multimea de antrenare
     plt.ylim((-0.5, 1.5))
     plt.xlim((-0.5, 1.5))
     xx = np.random.normal(0, 1, (100000))
     yy = np.random.normal(0, 1, (100000))
     X = np.array([xx, yy]).transpose()
     X = np.concatenate((X, X_))
     _, _, _, output = fwd(X, W_1, b_1, W_2, b_2)
     y = np.squeeze(np.round(output))
     plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
     plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
     plt.show(block=False)
     plt.pause(0.1)


def compute_y(x, W, bias):
     # dreapta de decizie
     # [x, y] * [W[0], W[1]] + b = 0
     return (-x*W[0] - bias) / (W[1] + 1e-10)



X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([-1,1,1,1])
epochs = 70
lr = 0.1
W,b,accuracy = train_perceptron(X,Y, epochs, lr)
print(W)
print(b)

print("exercitiul 3")
Y_3 = np.array([-1,1,1,-1])
W_3,b_3,accuracy_3 = train_perceptron(X,Y_3,epochs,lr)
print(W_3)
print(b_3)
print(accuracy_3)

#4

x_4 = np.array([[0, 0], [0, 1], [1,0], [1,1]])
y_4 = np.expand_dims(np.array([0, 1, 1, 0]), 1) # [[0], [1], ..]

# init no of neurons
no_hidden = 5
no_out = 1
# init weights
W_1 = np.random.normal(0, 1, (2, no_hidden))
b_1 = np.zeros((no_hidden))
W_2 = np.random.normal(0, 1, (no_hidden, no_out))
b_2 = np.zeros((no_out))
n = x_4.shape[0]
epochs = 70
lr = 0.5

for e in range(epochs):
  x_4, y_4 = shuffle(x_4, y_4)
  z_1, a_1, z_2, a_2 = fwd(x_4, W_1, W_2, b_1, b_2)
  Loss = -(y_4 * np.log(a_2) + (1 - y_4) * np.log(1 - a_2)).mean()
  print("Loss: ")
  print(Loss)
  dw_1, db_1, dw_2, db_2 = backward(a_1,a_2,z_1,W_2,x_4,y_4,num_samples=0)
  W_1 -= lr * dw_1  # lr - rata de invatare (learning rate)
  b_1 -= lr * db_1
  W_2 -= lr * dw_2
  b_2 -= lr * db_2
  plot_decision(x_4, W_1, W_2, b_1, b_2)








