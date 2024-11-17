import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import FakeData

import copy
import numpy as np

def get_model_diff(model1_state, model2_state):
    diff_state = {}
    for key in model1_state:
        diff_state[key] = model1_state[key] - model2_state[key]
    return diff_state

def plugin(w_locals, global_net, args):
    score = torch.zeros([args.num_clients, args.num_clients]).to(args.device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

    virtual_dataset = FakeData(size=args.ds, image_size=(3, 28, 28), num_classes=args.num_classes, transform=transform)
    virtual_loader = DataLoader(virtual_dataset, batch_size=args.bs, shuffle=True)

    net1 = resnet18(num_classes=args.num_classes).to(args.device)
    net2 = copy.deepcopy(net1).to(args.device)
    global_state = global_net.state_dict()
    cos = torch.nn.CosineSimilarity(dim=-1)

    for i, w_i in enumerate(w_locals):
        # Calculate w_i - global_model
        diff_state_1 = get_model_diff(w_i, global_state)
        net1.load_state_dict(diff_state_1)

        for batch, (virtual_data, _) in enumerate(virtual_loader):
            virtual_data = virtual_data.to(args.device)
            batch_score = torch.zeros([args.num_clients, args.num_clients]).to(args.device)

            # Get features from first difference model
            features1 = nn.Sequential(*list(net1.children())[:-1])(virtual_data).squeeze()

            for j, w_j in enumerate(w_locals[i:]):
                # Calculate w_j - global_model
                diff_state_2 = get_model_diff(w_j, global_state)
                net2.load_state_dict(diff_state_2)
                # Get features from second difference model
                features2 = nn.Sequential(*list(net2.children())[:-1])(virtual_data).squeeze()
                # Store the score symmetrically
                batch_score[i][j + i] = batch_score[j +i][i] = torch.sum(cos(features1, features2))

            # Accumulate batch scores
            score += batch_score

    # Calculate average score across all batches
    score = score / len(virtual_loader)

    return score
