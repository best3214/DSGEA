import numpy as np
import torch


# reverse the relationship
def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


# obtain training batches
def get_train_batch(x1, x2, train_set, k=5):
    # KG1
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]

    # KG2
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]

    # KG1 + KG2
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch


# calculate the hit rate
def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)

    # calculate distance matrix
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    hits1_left = 0
    # hits@1 and hits@10
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        if k == 1:
            hits1_left = Hk
        else:
            hits10_left = Hk

    # obtain the ranking of correct entity in the results
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    # calculate MRR according to rank
    MRR = (1/(rank+1)).mean().item()

    return hits1_left, hits10_left, MRR


# compute the complement of two set
def generate_other_entities(all_entities, part_entities):
    device = all_entities.device
    all_entities = all_entities.cpu().numpy()
    part_entities = part_entities.cpu().numpy()

    intersection = np.intersect1d(all_entities, part_entities)
    other_entities = np.setdiff1d(all_entities, intersection)

    return torch.tensor(other_entities).to(device)

# compute the intersection of two set
def intersect_nx2(tensor_a, tensor_b):
    # 将两个张量转换为集合
    set_a = set(map(tuple, tensor_a.tolist()))
    set_b = set(map(tuple, tensor_b.tolist()))

    # 求交集
    intersection_set = set_a.intersection(set_b)

    # 将交集转换回张量
    intersection_tensor = torch.tensor(list(intersection_set))

    return intersection_tensor





