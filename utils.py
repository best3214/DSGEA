import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    # print(train_batch)
    return train_batch

def get_sim_train_batch(cos_sim_mtx, train_set, k=5):
    # cos_sim_mtx = torch.matmul(sim_x1, sim_x2.t())

    pair1 = train_set[:, 0]
    pair2 = train_set[:, 1]

    sim_mtx = cos_sim_mtx.index_select(0, pair1)
    sim_mtx_inv = cos_sim_mtx.index_select(1, pair2).t()
    # print("sim_mtx.shape, sim_mtx_inv.shape:", sim_mtx.shape, sim_mtx_inv.shape)

    # pair_sim = torch.zeros(train_set.shape[0], device=cos_sim_mtx.device)
    # for i in range(pair.shape[0]):
    #     e2 = pair2[i]
    #     pair_sim[i] = sim_mtx[i][e2]
    # pair_sim = pair_sim.unsqueeze(1)
    topk_sim_index = sim_mtx.topk(k, largest=True)[1]
    topk_sim_index_inv = sim_mtx_inv.topk(k, largest=True)[1]
    return topk_sim_index, topk_sim_index_inv


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t',end='')
    hits1_left = 0
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        if k == 1:
            hits1_left = Hk
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t',end='')
    for k in Hn_nums:
        pred_topk= S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
    return hits1_left


def generate_other_entities(all_entities, part_entities):
    device = all_entities.device
    print("device:", device)
    # other_entities = torch.setdiff1d(all_entities, part_entities)
    all_entities = all_entities.cpu().numpy()
    part_entities = part_entities.cpu().numpy()

    intersection = np.intersect1d(all_entities, part_entities)
    other_entities = np.setdiff1d(all_entities, intersection)

    return torch.tensor(other_entities).to(device)

def generate_most_sim_pair(x1:torch.tensor, x2:torch.tensor, pair:torch.tensor, num=1000):
    dist = []
    for e1, e2 in pair:
        emb1 = x1[e1]
        emb2 = x2[e2]
        dis = torch.sum(torch.abs(emb1-emb2))
        dist.append(dis)
    dist = torch.tensor(dist)
    most_sim_indices = torch.sort(dist, descending=False)[1][:num]
    most_sim_pair = pair[most_sim_indices]
    return most_sim_pair


def generate_topk_sim_dict(x1: torch.tensor, x2: torch.tensor, other_entities1: torch.tensor,
                           other_entities2: torch.tensor, num=50):
    emb1 = x1[other_entities1]
    emb2 = x2[other_entities2]
    S = torch.cdist(emb1, emb2, p=1)

    topk_dict = dict()

    for i in range(S.shape[0]):
        topk_index = torch.topk(S[i], num, largest=False)[1]  # 获取每行最小的单个元素的索引
        topk_dict[other_entities1[i].item()] = np.array(other_entities2[topk_index])

    return topk_dict

def intersect_nx2(tensor_a, tensor_b):
    # 将两个张量转换为集合
    set_a = set(map(tuple, tensor_a.tolist()))
    set_b = set(map(tuple, tensor_b.tolist()))

    # 求交集
    intersection_set = set_a.intersection(set_b)

    # 将交集转换回张量
    intersection_tensor = torch.tensor(list(intersection_set))

    return intersection_tensor





