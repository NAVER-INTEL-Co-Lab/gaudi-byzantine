import copy

import torch
from torchvision.models import resnet18

from utils.test import test_img
from src.aggregation import server_avg

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)

def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)
    
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(args.device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors

def single_vectorization(w_glob, args):
    vector = copy.deepcopy(w_glob)
    for name in vector:
        vector[name] = vector[name].reshape([-1]).to(args.device)

    return torch.cat(list(vector.values()))

def pairwise_distance(w_locals, args):
    
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)
    
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                
    return distance    

def krum(w_locals, c, args):    
    n = len(w_locals) - c
    
    distance = pairwise_distance(w_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]

    chosen_idx = int(sorted_idx[0])
        
    return copy.deepcopy(w_locals[chosen_idx]), chosen_idx

def trimmed_mean(w_locals, c, args):
    n = len(w_locals) - 2 * c

    distance = pairwise_distance(w_locals, args)

    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]
        
    return server_avg([copy.deepcopy(w_locals[int(i)]) for i in chosen])

def fang(w_locals, dataset_val, c, args):
    
    loss_impact = {}
    net_a = resnet18(num_classes = args.num_classes)
    net_b = copy.deepcopy(net_a)

    for i in range(len(w_locals)):
        tmp_w_locals = copy.deepcopy(w_locals)
        w_a = trimmed_mean(tmp_w_locals, c, args)
        tmp_w_locals.pop(i)
        w_b = trimmed_mean(tmp_w_locals, c, args)
        
        net_a.load_state_dict(w_a)
        net_b.load_state_dict(w_b)
        
        _, loss_a = test_img(net_a.to(args.device), dataset_val, args)
        _, loss_b = test_img(net_b.to(args.device), dataset_val, args)
        
        loss_impact.update({i : loss_a - loss_b})
    
    sorted_loss_impact = sorted(loss_impact.items(), key = lambda item: item[1])
    filterd_clients = [sorted_loss_impact[i][0] for i in range(len(w_locals) - c)]

    return server_avg([copy.deepcopy(w_locals[i]) for i in filterd_clients])
    
    

    