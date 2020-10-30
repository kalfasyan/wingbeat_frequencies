import argparse
import os

import joblib
import libmr
import numpy as np
import scipy.spatial.distance as spd


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, _, model = query_weibull(category_name, weibull_model, distance_type) # dist was unused variable _
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob

def compute_channel_distances(mavs, features, eu_weight):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}

def calc_dists(mavs,scores, eu_weight=5e-3):
    return [compute_channel_distances(mcv, score, eu_weight) for mcv, score in zip(mavs, scores)]


def compute_mavs_and_scores(penulfeats, targets, nb_classes=3):
    """
    penulfeats: needs to be a layer of the same length as the nb_classes
    """

    scores = [[] for _ in range(nb_classes)]

    for score, t in zip (penulfeats, targets):
        if np.argmax(score) == t:
            scores[t].append(score)

    # Add channel axis (needed at multi-crop evaluation) # TODO: check if I need this
    scores = [np.array(x)[:, np.newaxis, :] for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)

    return mavs, scores


def evaluate_openmax(mavs, dists, scores, labels, distance_type, eu_weight=5e-3):
    from sklearn.metrics import accuracy_score, f1_score
    categories = np.unique(labels)

    tail_best, alpha_best, th_best = None, None, None
    f1_best = 0.0
    for tailsize in [20, 40, 80]:
        weibull_model = fit_weibull(mavs, dists, categories, tailsize, distance_type)
        for alpha in [3]:
            for th in [0.0, 0.5, 0.75, 0.8, 0.85, 0.9]:
                print(tailsize, alpha, th)
                pred_y, pred_y_o = [], []
                for score in scores:
                    so, ss = openmax(weibull_model, categories, score,
                                     eu_weight, alpha, distance_type)
                    pred_y.append(np.argmax(ss) if np.max(ss) >= th else 10)
                    pred_y_o.append(np.argmax(so) if np.max(so) >= th else 10)

                print(accuracy_score(labels, pred_y), accuracy_score(labels, pred_y_o))
                openmax_score = f1_score(labels, pred_y_o, average="macro")
                print(f1_score(labels, pred_y, average="macro"), openmax_score)
                if openmax_score > f1_best:
                    tail_best, alpha_best, th_best = tailsize, alpha, th
                    f1_best = openmax_score

    print("Best params:")
    print(tail_best, alpha_best, th_best, f1_best)
