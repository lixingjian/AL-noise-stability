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
from torch.autograd import Variable
import time

import copy
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def predict_prob_embed(models, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # FIXME: f'cuda:{CUDA_VISIBLE_DEVICES}'
    models['backbone'].eval()
    embDim = models['backbone'].get_embedding_dim()
    embeddings = None
    probs = None
    labeled_y = None

    with torch.no_grad():
        idx = 0
        for inputs, labels, _ in dataloader:
            x, y = inputs.to(device), labels.to(device)
            cout, emb, _ = models['backbone'](x)
            emb = emb.data.to(device)
            cout = cout.data.to(device)
            if embeddings is None:
                probs = cout
                embeddings = emb
                labeled_y = labels
            else:
                probs = torch.cat((probs, cout), dim=0)
                embeddings = torch.cat((embeddings, emb), dim=0)
                labeled_y = torch.cat((labeled_y, labels), dim=0)
    return probs, embeddings, labeled_y


def find_alphas(lb_embeddings, lb_labels, ulb_embeddings, ulb_preds, model, epsilon, args, grads=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # FIXME: f'cuda:{CUDA_VISIBLE_DEVICES}'

    unlabeled_size, embedding_size = ulb_embeddings.shape
    min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float32).to(device)
    pred_change = torch.zeros(unlabeled_size, dtype=torch.bool).to(device)

    ulb_embeddings = ulb_embeddings.to(device)

    if grads is not None:
        epsilon /= np.sqrt(embedding_size)
        grads = grads.to(device)

    for i in range(model.n_classes):
        emb = lb_embeddings[lb_labels == i]
        if emb.size(0) == 0:
            emb = lb_embeddings
        anchor = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1).to(device)

        # Find the maximum alpha for every unlabeled sample w.r.t the anchor
        if args.closed_form:
            # apply formula
            alpha = calculate_optimum_alpha(epsilon, anchor, ulb_embeddings, grads)
            mixed_embeddings = (1 - alpha) * ulb_embeddings + alpha * anchor
            alpha = alpha.to(device)
            out = model.forward_embedding(mixed_embeddings).detach()
            out = out.to(device)
            pred_change_mask = out.argmax(dim=1) != ulb_preds

        else:
            # calculate alpha by Gradient Descent
            alpha = init_random_alpha(ulb_embeddings.shape, epsilon)
            if args.alpha_opt:
                alpha, pred_change_mask = learn_alpha(ulb_embeddings, ulb_preds, model, anchor, alpha, epsilon, args)
            else:
                mixed_embeddings = (1 - alpha) * ulb_embeddings + alpha * anchor
                out = model.forward_embedding(mixed_embeddings).detach()
                out = out.to(device)
                alpha = alpha.to(device)
                pred_change_mask = out.argmax(dim=1) != ulb_preds

        # Update min_alphas and pred_change
        torch.cuda.empty_cache()
        alpha[~pred_change_mask] = 1
        pred_change[pred_change_mask] = True
        is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
        min_alphas[is_min] = alpha[is_min]

    return pred_change, min_alphas


def calculate_optimum_alpha(epsilon, anchor_embeddings, ulb_embeddings, ulb_grads):
    z = anchor_embeddings - ulb_embeddings
    alpha = (epsilon * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)
    return alpha


def init_random_alpha(size, epsilon):
    alpha = torch.normal(mean=epsilon/2, std=epsilon/2, size=size)
    alpha[torch.isnan(alpha)] = 1
    return torch.clamp(alpha, min=1e-8, max=epsilon)


def learn_alpha(ulb_embeddings, ulb_preds, model, anchor, alpha, epsilon, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # FIXME: f'cuda:{CUDA_VISIBLE_DEVICES}'

    ulb_preds = ulb_preds.to(device)
    min_alpha = torch.ones(alpha.size(), dtype=torch.float32).to(device)
    pred_changed = torch.zeros(ulb_preds.size(0), dtype=torch.bool).to(device)

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()

    for i in range(args.alpha_num_iters):
        for start_idx in range(0, alpha.size(0), args.alpha_batch_size):
            idx_range = slice(start_idx, start_idx + args.alpha_batch_size)

            model.zero_grad()
            batch_alpha = alpha[idx_range].to(device)
            batch_alpha_var = torch.autograd.Variable(batch_alpha, requires_grad=True)
            ulb_embed = ulb_embeddings[idx_range].to(device)
            ulb_pred = ulb_preds[idx_range].to(device)
            lb_anchor = anchor[idx_range].to(device)

            # Optimizer
            optimizer = torch.optim.Adam([batch_alpha_var],
                                         lr=args.alpha_lr / (1. if i < args.alpha_num_iters * 2 / 3 else 10.))

            # Forward
            embedding_mix = (1 - batch_alpha_var) * ulb_embed + batch_alpha_var * lb_anchor
            out = model.forward_embedding(embedding_mix)
            pred_change_mask = out.argmax(dim=1) != ulb_pred
            batch_loss = loss_func(out, ulb_pred)

            tmp_pred_change = torch.zeros(ulb_preds.size(0), dtype=torch.bool).to(device)
            tmp_alpha_change = torch.zeros(ulb_preds.size(0), dtype=torch.bool).to(device)

            tmp_pred_change[idx_range] = pred_change_mask
            pred_changed[idx_range] |= tmp_pred_change[idx_range].detach().to(device)

            l2_nrm = batch_alpha_var.norm(dim=1)
            tmp_alpha_change[idx_range] = tmp_pred_change[idx_range] & (l2_nrm < min_alpha[idx_range].norm(dim=1))
            min_alpha[tmp_alpha_change] = batch_alpha_var[tmp_alpha_change[idx_range]].detach().to(device)

            batch_loss *= -1

            loss = args.alpha_l2_coef * l2_nrm - args.alpha_clf_coef * batch_loss
            loss.sum().backward(retain_graph=True)
            optimizer.step()

            batch_alpha_var = torch.clamp(batch_alpha_var, min=1e-8, max=epsilon)
            alpha[idx_range] = batch_alpha_var.detach().to(device)

            del batch_alpha_var, ulb_embed, lb_anchor, embedding_mix
            torch.cuda.empty_cache()

    return min_alpha.to(device), pred_changed.to(device)


def sample_from_candidates(features, k):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    cluster_indices = kmeans.fit_predict(features)

    # Get the closest sample to the center for each cluster
    cluster_centers = kmeans.cluster_centers_[cluster_indices]
    dist_to_center = np.linalg.norm(features - cluster_centers, axis=1)
    org_indices = np.arange(features.shape[0])
    return np.array([org_indices[cluster_indices == i][dist_to_center[cluster_indices == i].argmin()]
                     for i in range(k) if (cluster_indices == i).sum() > 0])


def feature_mix_sampling(models, unlabeled_loader, labeled_loader, args, idxs_prohibited=None):
    n_sampled = ADDENDUM
    model = models['backbone']

    if idxs_prohibited is not None:
        raise NotImplementedError('idxs_prohibited')

    # debug(f'Unlabeled batches: {len(unlabeled_loader)}, currently labeled: {len(labeled_loader)}')
    # debug(f'Trying to select {n_sampled} samples')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # FIXME: f'cuda:{CUDA_VISIBLE_DEVICES}'

    # Get predictions, probabilities and representations
    _, lb_embeddings, lb_labels = predict_prob_embed(models, labeled_loader)
    ulb_probs, ulb_embeddings, _ = predict_prob_embed(models, unlabeled_loader)
    ulb_preds = ulb_probs.argmax(dim=1)

    # Set grads for closed_form formula
    grads = None
    if args.closed_form:
        var_emb = Variable(ulb_embeddings, requires_grad=True).to(device)  # cuda()
        out = model.forward_embedding(var_emb)
        loss = F.cross_entropy(out, ulb_preds.to(device))  # cuda()
        grads = torch.autograd.grad(loss, var_emb)[0].data.to(device)
        del loss, var_emb, out

    # Choose the optimum epsilon: maximum norm of alpha
    unlabeled_size, embedding_size = ulb_embeddings.shape
    min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float32).to(device)
    candidates = torch.zeros(unlabeled_size, dtype=torch.bool).to(device)
    epsilon = 0

    while epsilon < 1 and candidates.sum() < n_sampled:
        epsilon += args.epsilon

        tmp_pred_change, tmp_min_alphas = find_alphas(lb_embeddings, lb_labels,
                                                      ulb_embeddings, ulb_preds,
                                                      model, epsilon, args, grads)
        min_alpha_changed = min_alphas.norm(dim=1) > tmp_min_alphas.norm(dim=1)
        min_alphas[min_alpha_changed] = tmp_min_alphas[min_alpha_changed]
        candidates |= tmp_pred_change

        # debug(f'(loop) epsilon={epsilon}, {int(candidates.sum().item())} inconsistencies in all')

    debug(f'Epsilon: {epsilon}')
    debug(f'Candidate set size: {candidates.sum().item()}, trying to select {n_sampled} of them')

    # Get samples from the candidate set
    sample_indices = np.array(list(), dtype=int)
    if candidates.sum() > 0:
        candidate_features = F.normalize(ulb_embeddings[candidates]).detach().cpu()
        samples = sample_from_candidates(candidate_features, min(n_sampled, candidates.sum().item()))
        sample_indices = candidates.nonzero(as_tuple=True)[0][samples].cpu()

    debug(f'Samples: {sample_indices.tolist()}')

    # Fill empty places by random choice
    if len(sample_indices) < n_sampled:
        remaining = n_sampled - len(sample_indices)
        ulb_selected = np.zeros(unlabeled_size, dtype=bool)
        ulb_selected[sample_indices] = True
        ulb_unselected = np.where(ulb_selected == False)[0]
        sample_indices = np.concatenate((sample_indices, np.random.choice(ulb_unselected, remaining)))
        debug(f'Not enough candidates, filling empty spots: {sample_indices.tolist()}')

    uncertainty = torch.zeros(len(unlabeled_loader.sampler))
    print(uncertainty.shape)
    uncertainty[sample_indices] = 1
    return uncertainty


def debug(*args, **kwargs):
    print('[DEBUG - AlphaMix] ', end='')
    print(*args, **kwargs)
