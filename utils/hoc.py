# HOC estimator

import numpy as np
import torch
import time
from tqdm import tqdm
from torch.nn import functional as F


smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)

"""
Adapted from the docta.ai hoc.py. Removed the dependency on cfg.
Original: https://github.com/Docta-ai/docta/blob/master/docta/core/hoc.py
"""

def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1)*1e2 # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = distance_matrix.cpu().float()/1e4
    return distance_matrix

def consensus_analytical(num_classes, T, P, mode):
    r""" Compute the first-, second-, and third-order of consensus matrices.
    Args:
        cfg: configuration
        T : noise transition matrix
        P : the priors of P(Y = i), i \in [K]
        mode :
    Return:
        c_analytical[0] : first-order consensus
        c_analytical[1] : second-order consensus
        c_analytical[2] : third-order consensus 
    """

    KINDS = num_classes
    P = P.reshape((KINDS, 1))
    c_analytical = [[] for _ in range(3)]

    c_analytical[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    c_analytical[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        c_analytical[1] = torch.cat(
            [c_analytical[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with c_est
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                c_analytical[2][r][(i+r+KINDS) % KINDS] = temp33[r]
        else:
            c_analytical[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]

    # adjust the order of the output (N*N), keeping consistent with c_est
    temp = []
    for p1 in range(KINDS):
        temp = torch.cat(
            (c_analytical[1][p1, KINDS-p1:], c_analytical[1][p1, :KINDS-p1]))
        c_analytical[1][p1] = temp
    return c_analytical


def func(hoc_cfg, c_est, T_out, P_out, num_classes):
    """ Compute the loss of estimated consensus matrices
    Example hoc_cfg:
                hoc_cfg = dict(
                max_step = 1501, 
                T0 = None, 
                p0 = None, 
                lr = 0.1, 
                num_rounds = 50, 
                sample_size = 35000, # 70% of dataset, ref: https://github.com/Docta-ai/docta/issues/5
                already_2nn = False,
                device = 'cpu'
            )

    """
    loss = torch.tensor(0.0).to(hoc_cfg.device)  # initialize the loss

    P = smp(P_out)
    T = smt(T_out)

    # mode = random.randint(0, cfg.num_classes - 1) # random update for speedup
    mode = -1  # calculate all patterns

    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at
    # this time: N, N*N, N*N*N
    c_ana = consensus_analytical(
        num_classes, T.to(hoc_cfg.device), P.to(hoc_cfg.device), mode)

    # weight for differet orders of consensus patterns
    weight = [1.0, 1.0, 1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        c_ana[j] = c_ana[j].to(hoc_cfg.device)
        loss += weight[j] * torch.norm(c_est[j] - c_ana[j])  # / np.sqrt(N**j)

    return loss


def calc_func(hoc_cfg, num_classes , c_est, show_details=True):
    """ Optimize over the noise transition matrix T and prior P
    """

    N = num_classes
    hoc_cfg.device = torch.device(hoc_cfg.device)
    if hoc_cfg.T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N, N))
    else:
        T = hoc_cfg.T0

    if hoc_cfg.p0 is None:
        P = torch.ones((N, 1)) / N + torch.rand((N, 1)) * 0.1
    else:
        P = hoc_cfg.p0

    T = T.to(hoc_cfg.device)
    P = P.to(hoc_cfg.device)
    c_est = [item.to(hoc_cfg.device) for item in c_est]
    print(f'Use {hoc_cfg.device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr=hoc_cfg.lr)

    # train
    loss_min = 100.0
    T_rec = T.detach()
    P_rec = P.detach()

    time1 = time.time()
    # use gradient descent to solve consensus equations
    for step in tqdm(range(hoc_cfg.max_step)):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(hoc_cfg, c_est, T, P, num_classes)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()

        if show_details:  # print log
            if step % 100 == 0:
                print('loss {}'.format(loss))
                print(f'step: {step}  time_cost: {time.time() - time1}')
                print(
                    f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
                print(
                    f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
                time1 = time.time()
        # release cuda cache
        torch.cuda.empty_cache()
    print(f'Solve equations... [Done]')
    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def get_consensus_patterns(dataset, sample, device, k=1+2):
    """ KNN estimation
    """
    feature = dataset.feature if isinstance(
        dataset.feature, torch.Tensor) else torch.tensor(dataset.feature)
    label = dataset.label if isinstance(
        dataset.label, torch.Tensor) else torch.tensor(dataset.label)
    feature = feature[sample]
    label = label[sample]
    feature = feature.to(torch.float16).to(device)
    dist = cosDistance(feature)
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    knn_labels = label[indices]
    return knn_labels, values


def consensus_counts(num_classes, consensus_patterns):
    """ Count the consensus
    """
    KINDS = num_classes

    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)

    for _, pattern in enumerate(consensus_patterns):
        cnt[0][pattern[0]] += 1
        cnt[1][pattern[0]][pattern[1]] += 1
        cnt[2][pattern[0]][pattern[1]][pattern[2]] += 1

    return cnt


def estimator_hoc(num_classes, hoc_cfg, dataset, show_details=True):
    """ HOC estimator
    """
    print('Estimating consensus patterns...')

    KINDS = num_classes

    # initialize sample counts
    c_est = [[] for _ in range(3)]
    c_est[0] = torch.zeros(KINDS)
    c_est[1] = torch.zeros(KINDS, KINDS)
    c_est[2] = torch.zeros(KINDS, KINDS, KINDS)

    sample_size = int(len(dataset) * 0.9)

    if hoc_cfg is not None and hoc_cfg.sample_size:
        sample_size = np.min((hoc_cfg.sample_size, sample_size))

    for idx in tqdm(range(hoc_cfg.num_rounds)):
        if show_details:
            print(idx, flush=True)

        sample = np.random.choice(
            range(len(dataset)), sample_size, replace=False)

        if not hoc_cfg.already_2nn:
            consensus_patterns_sample, _ = get_consensus_patterns(
                dataset, sample, hoc_cfg.device)
        else:
            consensus_patterns_sample = torch.tensor(dataset.consensus_patterns)[sample] if isinstance(
                dataset.consensus_patterns, list) else dataset.consensus_patterns[sample]
        cnt_y_3 = consensus_counts(num_classes, consensus_patterns_sample)
        for i in range(3):
            cnt_y_3[i] /= sample_size
            c_est[i] = c_est[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
        # release cuda cache
        torch.cuda.empty_cache()

    print('Estimating consensus patterns... [Done]')

    for j in range(3):
        c_est[j] = c_est[j] / hoc_cfg.num_rounds

    _, T_est, p_est, T_est_before_sfmx = calc_func(hoc_cfg=hoc_cfg, num_classes=num_classes, c_est=c_est, show_details=show_details)

    T_est = T_est.cpu().numpy()
    T_est_before_sfmx = T_est_before_sfmx.cpu().numpy()
    p_est = p_est.cpu().numpy()
    return T_est, p_est, T_est_before_sfmx
