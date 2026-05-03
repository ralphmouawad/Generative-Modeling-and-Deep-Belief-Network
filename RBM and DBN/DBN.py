from RBM import RBM
import numpy as np
import matplotlib.pyplot as plt

class DBN:
  def __init__(self):
    self.RBM_layers = [] ## DBN as a list of RBMs

  def init_DBN(self, layer_sizes):
    for i in range(len(layer_sizes) - 1):
      rbm = RBM(layer_sizes[i], layer_sizes[i+1])
      self.RBM_layers.append(rbm)
  
  def train_DBN(self, X, iter, lr, batch_size):
    for rbm in self.RBM_layers:
      rbm.train_RBM(X, lr, batch_size, iter)
      X = rbm.entree_sortie(X)
  
  def generer_image_DBN(self, nb_images, iter_Gibbs):
    ## here we should use the last RBM layer 
    vis_size = len(self.RBM_layers[-1].a)
    gen_images = self.RBM_layers[-1].generer_image_RBM(nb_images, iter_Gibbs, (vis_size,))
    for i in range(len(self.RBM_layers) - 2, -1, -1):
      gen_images = (
          np.random.rand(nb_images, len(self.RBM_layers[i].a)) < self.RBM_layers[i].sortie_entree(gen_images)) * 1
    return gen_images