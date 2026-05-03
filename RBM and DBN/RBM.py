# Restricted Boltzmann Machines

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class RBM:
  def __init__(self, p, q):
    self.W = 0.1 * np.random.randn(p, q)
    self.a = np.zeros(p)
    self.b = np.zeros(q)

  def sigmoid(self, x):
    return 1/ (1 + np.exp(-x))

  def entree_sortie(self, x):
    return self.sigmoid(x @ self.W + self.b)

  def sortie_entree(self, H):
    return self.sigmoid(H @ self.W.T + self.a)

  def train_RBM(self, X, lr, batch_size, epoch):
    p, q = self.W.shape
    weights_history = []
    losses = []
    n = np.size(X, 0)

    for iter in range(epoch):
      perm = np.random.permutation(n)
      X = X[perm]
      for i in range(0, n, batch_size):
        X_batch = X[i:min(i + batch_size, n)]
        t_batch = np.size(X_batch, 0)

        v_0 = X_batch
        p_h_v_0 = self.entree_sortie(v_0)
        h_0 = (np.random.rand(t_batch, q) <= p_h_v_0) * 1
        p_v_h_0 = self.sortie_entree(h_0)
        v_1 = (np.random.rand(t_batch, p) <= p_v_h_0) * 1
        p_h_v_1 = self.entree_sortie(v_1)

        d_a = np.sum(v_0 - v_1, axis=0)
        d_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
        d_W = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

        self.a += lr * d_a
        self.b += lr * d_b
        self.W += lr * d_W

        weights_history.append(np.mean(self.W))

      H = self.entree_sortie(X)
      X_rec = self.sortie_entree(H)
      err = np.mean((X - X_rec) ** 2)
      losses.append(err)
      if iter % 10 == 0:
        print(f"Epoch: {iter}, Error: {err}")

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss vs Epochs')
    plt.show()
    print('Final L2 Loss:', losses[-1])

    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.title('Weights vs Epochs')
    plt.plot(weights_history)
    plt.show()

  def generer_image_RBM(self, nb_images, iter_Gibbs, image_size):
    p, q = self.W.shape
    images = []
    for _ in range(nb_images):
      V = (np.random.rand(p) < 0.5) * 1
      for _ in range(iter_Gibbs):
        h = (np.random.rand(q) < self.entree_sortie(V)) * 1
        V = (np.random.rand(p) < self.sortie_entree(h)) * 1
      images.append(V.reshape(image_size))
    return images

