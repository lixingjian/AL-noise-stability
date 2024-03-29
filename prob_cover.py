import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
from config import *
from query_models import VAE, Discriminator, GCN
from sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
import torch.nn.functional as F 
from torch.distributions import Categorical
import time

import copy
from scipy import stats
from sklearn.metrics import pairwise_distances

def predict_prob_embed(models, dataloader):
    models['backbone'].eval()
    embDim = models['backbone'].get_embedding_dim()
    embeddings = None
    probs = None

    with torch.no_grad():
        idx = 0
        for inputs, labels, _ in dataloader:
            x, y = inputs.cuda(), labels.cuda()
            cout, emb, _ = models['backbone'](x)
            emb = emb.data.cpu()
            cout = cout.data.cpu()
            if embeddings is None:
                embeddings = emb
                probs = cout
            else:
                probs = torch.cat((probs, cout), dim = 0)
                embeddings = torch.cat((embeddings, emb), dim = 0)
    return probs, embeddings

import numpy as np
import pandas as pd
import torch
import random

class ProbCover:
    def __init__(self, models, dataloader, lSet, uSet, delta):
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = ADDENDUM
        self.delta = delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        _, features = predict_prob_embed(models, dataloader)
        features = features.detach().numpy()
        self.rel_features = features / np.linalg.norm(features, axis=1, keepdims=True)
        print(self.relevant_indices.shape, self.rel_features.shape)
        self.graph_df = self.construct_graph()

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.
        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            #print(dist.max(), dist)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)
        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.budgetSize):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            if degrees.max() > 0:
                cur = degrees.argmax()
            else:
                cur = random.randint(0, SUBSET - 1)
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}\tCur is {cur}')
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return selected, activeSet, remainSet

def prob_cover_sampling(models, dataloader, lSet, uSet, args):
    pbcover = ProbCover(models, dataloader, lSet, uSet, 0.4)
    chosen, _, _ = pbcover.select_samples()
    print('chosen:', sorted(chosen))
    arg = [0] * (SUBSET - ADDENDUM) + chosen
    assert len(arg) == SUBSET
    return np.array(arg)
