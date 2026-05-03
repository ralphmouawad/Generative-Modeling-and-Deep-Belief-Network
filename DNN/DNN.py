import numpy as np
from RBM import RBM
from DBN import DBN


class DNN():
    def __init__(self):
        self.DBN = DBN()
        # self.classification = None

    def init_DNN(self, layers_size, classes):
        self.DBN.init_DBN(layers_size)
        self.classification = RBM(layers_size[-1], classes)

    def pretrain_DNN(self, data, iter, lr, batch_size):
        self.DBN.train_DBN(data, iter, lr, batch_size)

    def calcul_softmax(self, classif, data):
        out = np.dot(data, classif.W) + classif.b
        return np.exp(out) / np.sum(np.exp(out), axis = 1, keepdims= True)

    def entree_sortie_reseau(self, data):
        activation = [data]
        output = data

        for lay in self.DBN.RBM_layers:
            output = lay.entree_sortie(output)
            activation.append(output)

        probs = self.calcul_softmax(self.classification, output)

        return activation, probs

    def retropropagation(self, iter, lr, batch_size, data, labels):
        n = data.shape[0]
        for epoch in range(iter):

            # Shuffle the data for stable learning
            perm = np.random.permutation(n)
            data = data[perm]
            labels = labels[perm]
            for i in range(0, data.shape[0], batch_size):
                    
                batch = data[i: min(i+batch_size, n)]
                label = labels[i: min(i+batch_size, n)]

                current_batch_size = batch.shape[0]
                
                # Get last layer softmax
                values, probs = self.entree_sortie_reseau(batch)

                # Get the errors
                erreur = probs - label

                # Get the gradients
                dW = np.dot(values[-1].T, erreur) / current_batch_size
                db = np.sum(erreur, axis = 0) / current_batch_size

                # Update classification layer weights
                W_prev = self.classification.W.copy()
                self.classification.W -= lr * dW
                self.classification.b -= lr * db

                da = np.dot(erreur, W_prev.T)
                for j in range(len(self.DBN.RBM_layers)-1, -1, -1) :
                    lay = self.DBN.RBM_layers[j]
                    sig = values[j+1]
                    sig_prev = values[j]

                    dz_lay = da * sig * (1 - sig)
                    dW_lay = np.dot(sig_prev.T, dz_lay) / current_batch_size
                    db_lay = np.sum(dz_lay, axis=0) / current_batch_size

                    W_lay_prev = lay.W.copy()
                    lay.W -= lr * dW_lay
                    lay.b -= lr * db_lay
                    da = np.dot(dz_lay, W_lay_prev.T)

            _, preds = self.entree_sortie_reseau(data)
            cross_entropy = - np.mean( np.sum(labels * np.log(preds + 1e-9), axis=1) )
            # if epoch % 10 == 0:
            print(f'Loss at epoch {epoch} : {cross_entropy:.3f}')


    def test_DNN(self, test_data, test_labels):
        _, probs = self.entree_sortie_reseau(test_data)
        pred_labels = np.argmax(probs, axis=1)
        err = np.sum(1*(test_labels != pred_labels))

        return  err / len(test_data)


