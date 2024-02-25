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
import backbone
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import copy


def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):  # and isinstance(m, nn.Conv2d):
            noise = torch.randn(m.weight.size())
            noise = noise.cpu()
            noise *= (NOISE_SCALE * m.weight.norm() / noise.norm())
            m.weight.add_(noise)
            # print('scale', 1.0 * noise.norm() / m.weight.norm(), 'weight', m.weight.view(-1)[:10])


def noise_stability_sampling(models, unlabeled_loader, args):
    if NOISE_SCALE < 1e-8:
        uncertainty = torch.randn(SUBSET)
        return uncertainty

    uncertainty = torch.zeros(SUBSET).cpu()

    diffs = torch.tensor([]).cpu()
    use_feature = args.dataset in ['house']
    outputs = get_all_outputs(models['backbone'], unlabeled_loader, use_feature)
    for i in range(args.n_sampling):
        noisy_model = copy.deepcopy(models['backbone'])
        noisy_model.eval()

        noisy_model.apply(add_noise_to_weights)
        outputs_noisy = get_all_outputs(noisy_model, unlabeled_loader, use_feature)

        diff_k = outputs_noisy - outputs
        for j in range(diff_k.shape[0]):
            diff_k[j, :] /= outputs[j].norm()
        diffs = torch.cat((diffs, diff_k), dim=1)

    indsAll, _ = kcenter_greedy(diffs, ADDENDUM)
    # print(indsAll)
    for ind in indsAll:
        uncertainty[ind] = 1

    return uncertainty.cpu()


from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear


def k_dpp(X, K):
    DPP = FiniteDPP('likelihood', **{'L_gram_factor': 1e6 * X.cpu().numpy().transpose()})
    DPP.flush_samples()
    DPP.sample_mcmc_k_dpp(size=K)
    indsAll = DPP.list_of_samples[0][0]
    return indsAll


def kcenter_greedy(X, K):
    if K <= 0:
        return list(), list()
    elif K >= X.shape[0]:
        return list(range(X.shape[0])), list(range(X.shape[0]))

    # avg_norm = np.mean([torch.norm(X[i]).item() for i in range(X.shape[0])])
    mu = torch.zeros(1, X.shape[1]).cpu()
    D2 = torch.norm(X, dim=1)
    nearestId = -torch.ones(X.shape[0], dtype=torch.long).cpu()
    indsAll = []
    while len(indsAll) < K:
        for i, ind in enumerate(D2.topk(1)[1]):
            # if i == 0:
            #     print(len(indsAll), ind.item(), D2[ind].item(), X[ind,:5])
            D2[ind] = 0
            nearestId[ind] = ind
            mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
            indsAll.append(ind)

        newD = torch.cdist(X, mu[-1:]).squeeze(1)
        less_D2_mask = (D2 > newD)
        D2[less_D2_mask] = newD[less_D2_mask]
        nearestId[less_D2_mask] = indsAll[-1]

    # selected_norm = np.mean([torch.norm(X[i]).item() for i in indsAll])

    return torch.tensor(indsAll), nearestId


def get_all_outputs(model, unlabeled_loader, use_feature=False):
    model.eval()
    outputs = torch.tensor([]).cpu()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cpu()
            out, fea, _ = model(inputs)
            if use_feature:
                out = fea
            else:
                out = F.softmax(out, dim=1)
            outputs = torch.cat((outputs, out), dim=0)

    return outputs

