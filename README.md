Part I - Neural Networks Pre-training:
- Built a stochastic shallow network using Restricted Boltzmann Machines (RBM) trained through the Contrastive Divergence (CD-1) algorithm and Markov Chain Monte Carlo (Gibbs Sampling).
- Built a Deep Belief Network (DBN) through a stack of RBMs and trained it using the Greedy-layer Wise Procedure algorithm.
- Built a Neural Networks: first pre-trained its weights as treating like a DBN followed by regular training with backpropagation. Compared its results on classification when only trained with backpropagation. Results showed that pre-training lets the model learn the structure of the data before seeing results.

Part II - Generative Modeling on MNIST:
- Generated images using RBM and DBN as described above.
- Trained Generative Adversarial Networks using a min-max training procedure.
- Trained Variational Autoencoders through variational inference and latent space representation.
- Trained a Denoising Diffusion Probabilistic Model by learning a the noise injected and sampled using the Unadjusted Langevin Algorithm (ULA).
- Trained a Score-Based Generative Model and sampled through Euler Maruyama time discretization of Stochastic Differential Equations.
- Trained and Sampled from Flow Matching by learning a velocity field and used Euler discretization of ODEs.
