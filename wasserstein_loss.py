import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def _pairwise_distance(x, y, squared=False, eps=1e-16):
	# Compute the 2D matrix of distances between all the embeddings.
	# got the dot product between all embeddings
	cor_mat = torch.matmul(x, y.t())

	# Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
	# This also provides more numerical stability (the diagonal of the result will be exactly 0).
	norm_mat = cor_mat.diag()

	# Compute the pairwise distance matrix as we have:
	# ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
	# shape (batch_size, batch_size)
	distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

	# Because of computation errors, some distances might be negative so we put everything >= 0.0
	distances = F.relu(distances)

	if not squared:
		# Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
		# we need to add a small epsilon where distances == 0.0
		mask = torch.eq(distances, 0.0).float()
		distances = distances + mask * eps
		distances = torch.sqrt(distances)

		# Correct the epsilon added: set the distances on the mask to be exactly 0.0
		distances = distances * (1.0 - mask)

	return distances


'''
E_Q[\phi^{c}_{\epsilion}(x)] + E_P[\phi(y)]

\phi^{c}_{\epsilion}(x) = -\epsilon * \log[ E_P[ \exp( (-d(x,y)+\phi(y)) : \epsilon ) ] ]

kan_network: \phi(y)
ot_cost: 	 d(x,y)
y: source domain  
'''

def entropic_wasserstein_loss(x, y, kantorovich_val, batch_size, epsilon=0.1):
	#import pdb; pdb.set_trace()
	ot_cost = _pairwise_distance(x, y)

	exp_term = (-ot_cost + kantorovich_val) / epsilon

	kan_network_loss = kantorovich_val.mean()

	ot_loss = - (torch.mean(- epsilon * (torch.log(torch.tensor(1.0 / batch_size)) + torch.logsumexp(exp_term, dim=1))) + kan_network_loss)

	return ot_loss



