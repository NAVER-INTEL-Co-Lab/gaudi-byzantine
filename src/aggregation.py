import copy

import torch

def server_avg(w_locals):
    w_avg = copy.deepcopy(w_locals[0])
    
    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]            
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            
    return w_avg

def server_dyn(args, global_model, w_locals, idx_users, controls, sorted_idxs):

    for idx_user in idx_users:
        if idx_user not in controls:
            controls[idx_user] = {k: torch.zeros_like(param, device=args.device) for k, param in global_model.state_dict().items()}

    w_avg = copy.deepcopy(w_locals[0])

    with torch.no_grad():
        # Perform weight averaging
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            if len(sorted_idxs):
                for i, idx_user in enumerate(sorted_idxs):
                    controls[idx_user][k] += global_model.state_dict()[k] - w_locals[i][k]
                    w_avg[k] -= args.mu * controls[idx_user][k]
            else:
                for i, idx_user in enumerate(idx_users):
                    controls[idx_user][k] += global_model.state_dict()[k] - w_locals[i][k]
                    w_avg[k] -= args.mu * controls[idx_user][k]
                
    # Update the global model with the averaged and regularized weights
    global_model.load_state_dict(w_avg)

    return global_model.state_dict(), controls