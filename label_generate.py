import math
import os
import torch
import numpy as np
from conf import parse_args
import os
from tqdm import trange, tqdm

from utils import generate_other_entities, generate_topk_sim_dict, generate_most_sim_pair, intersect_nx2

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = parse_args()

def enhance_latent_labels_with_OSS(x1, x2, improved_candi_probs: torch.Tensor,
                                                         improved_candi_probs_inv: torch.Tensor, data, ratio=0):
    print('enhance_latent_labels_with_OSS')
    device = improved_candi_probs.device
    ptrue = 0
    pfalse = 0

    new_alignment = []


    with torch.no_grad():
        pair_num = data.test_set.size(0)

        dis_mtx = torch.cdist(x1, x2)
        dis_mtx_inv = dis_mtx.t()

        all_entities1 = torch.arange(improved_candi_probs.shape[0]).to(device)
        all_entities2 = torch.arange(improved_candi_probs.shape[1]).to(device)

        other_entities1 = generate_other_entities(all_entities1, data.train_set[:, 0])
        other_entities2 = generate_other_entities(all_entities2, data.train_set[:, 1])

        new_alignment_dict1 = {}
        new_alignment_dict2 = {}

        for ent1, ent2 in tqdm(data.temp_set.cpu().numpy(), ascii=True, desc="Generating new alignment labels"):
            # print("ent1, ent2:", ent1, ent2)
            if ent1 in data.neigh1 and ent2 in data.neigh2:
                # print(ent1)
                neighbors1 = np.array(data.neigh1[ent1])
                neighbors2 = np.array(data.neigh2[ent2])
                neighbors1 = np.intersect1d(neighbors1, other_entities1.numpy())
                neighbors2 = np.intersect1d(neighbors2, other_entities2.numpy())
                if neighbors1.size > 0 and neighbors2.size > 0:
                    dist = dis_mtx[neighbors1]
                    dist = dist[:, neighbors2]
                    dist_inv = dis_mtx_inv[neighbors2]
                    dist_inv = dist_inv[:, neighbors1]
                    candi_probs = improved_candi_probs[neighbors1]
                    candi_probs = candi_probs[:, neighbors2]
                    candi_probs_inv = improved_candi_probs_inv[neighbors2]
                    candi_probs_inv = candi_probs_inv[:, neighbors1]
                    num1 = len(neighbors1)
                    for num in range(num1):
                        probs = candi_probs[num]
                        dis = dist[num]
                        # print(probs)
                        d, tmp_idx_dis = torch.min(dis, dim=-1, keepdim=False)
                        prob, tmp_idx = torch.max(probs, dim=-1, keepdim=False)
                        if tmp_idx_dis != tmp_idx:
                            continue
                        d = d.cpu().item()
                        prob = prob.cpu().item()
                        tmp_idx_dis = tmp_idx_dis.cpu().item()
                        tmp_idx = tmp_idx.cpu().item()
                        dis_inv = dist_inv[tmp_idx_dis]
                        d_inv, tmp_idx_dis_inv = torch.min(dis_inv, dim=-1, keepdim=False)

                        probs_inv = candi_probs_inv[tmp_idx]
                        prob_inv, tmp_idx_inv = torch.max(probs_inv, dim=-1, keepdim=False)
                        prob_inv = prob_inv.cpu().item()
                        tmp_idx_inv = tmp_idx_inv.cpu().item()

                        # if num == tmp_idx_inv:
                        if num == tmp_idx_inv and num == tmp_idx_dis_inv:
                            if prob > ratio and prob_inv > ratio:
                                if neighbors1[num] not in new_alignment_dict1 and neighbors2[tmp_idx] not in new_alignment_dict2:
                                # if neighbors1[num] not in new_alignment_dict1 and neighbors2[tmp_idx] not in new_alignment_dict2 and neighbors1[num] in topk_dict2[neighbors2[tmp_idx]] and neighbors2[tmp_idx] in topk_dict1[neighbors1[num]]:
                                    new_alignment_dict1[neighbors1[num]] = prob
                                    new_alignment_dict2[neighbors2[tmp_idx]] = prob_inv
                                elif neighbors1[num] in new_alignment_dict1:
                                    if new_alignment_dict1[neighbors1[num]] <= prob:
                                        value_to_remove = neighbors1[num]  # 要移除的值
                                        new_alignment = [row for row in new_alignment if row[0] != value_to_remove]
                                    else:
                                        continue
                                elif neighbors2[tmp_idx] in new_alignment_dict2:
                                    if new_alignment_dict2[neighbors2[tmp_idx]] <= prob_inv:
                                        value_to_remove = neighbors2[tmp_idx]  # 要移除的值
                                        new_alignment = [row for row in new_alignment if row[1] != value_to_remove]
                                    else:
                                        continue
                                else:
                                    continue

                                new_alignment.append([neighbors1[num], neighbors2[tmp_idx]])
        if data.new_set.shape[0] > 0:
            data.new_set = intersect_nx2(data.new_set.cpu(), torch.tensor(new_alignment)).to(data.train_set.device)
        else:
            data.new_set = torch.tensor(new_alignment, device=data.train_set.device)
        for i in data.train_set.cpu().numpy():
            new_alignment.append(i)

    for ent1, ent2 in data.train_set.cpu().numpy():
        if ent1 == ent2:
            ptrue += 1
        else:
            pfalse += 1
    rec_num = ptrue + pfalse
    if rec_num == 0:
        pacc = 0
    else:
        pacc = ptrue / (ptrue + pfalse)
    print("new labels num:", ptrue + pfalse)
    print("joint distr threshold & mutual nearest acc", pacc)
def enhance_latent_labels_with_DAA(x1, x2, improved_candi_probs: torch.Tensor,
                                                         improved_candi_probs_inv: torch.Tensor, data, ratio=0):
    print('enhance_latent_labels_with_DAA')
    device = improved_candi_probs.device
    train_alignment = data.train_set.cpu().numpy()
    test_alignment = data.test_set.cpu().numpy()


    # ratio = ratio
    print('ratio', ratio)
    ptrue = 0
    pfalse = 0

    new_alignment = []
    with torch.no_grad():
        pair_num = data.test_set.size(0)

        dis_mtx = torch.cdist(x1, x2)
        dis_mtx_inv = dis_mtx.t()

        all_entities1 = torch.arange(improved_candi_probs.shape[0]).to(device)
        all_entities2 = torch.arange(improved_candi_probs.shape[1]).to(device)

        other_entities1 = generate_other_entities(all_entities1, data.train_set[:, 0])
        other_entities2 = generate_other_entities(all_entities2, data.train_set[:, 1])

        new_alignment_dict1 = {}
        new_alignment_dict2 = {}


        for ent1, ent2 in tqdm(data.temp_set.cpu().numpy(), ascii=True, desc="Generating new alignment labels"):
            if ent1 in data.neigh1 and ent2 in data.neigh2:
                neighbors1 = np.array(data.neigh1[ent1])
                neighbors2 = np.array(data.neigh2[ent2])
                neighbors1 = np.intersect1d(neighbors1, other_entities1.numpy())
                neighbors2 = np.intersect1d(neighbors2, other_entities2.numpy())
                if neighbors1.size > 0 and neighbors2.size > 0:
                    dist = dis_mtx[neighbors1]
                    dist = dist[:, neighbors2]
                    dist_inv = dis_mtx_inv[neighbors2]
                    dist_inv = dist_inv[:, neighbors1]
                    candi_probs = improved_candi_probs[neighbors1]
                    candi_probs = candi_probs[:, neighbors2]
                    candi_probs_inv = improved_candi_probs_inv[neighbors2]
                    candi_probs_inv = candi_probs_inv[:, neighbors1]
                    num1 = len(neighbors1)
                    num2 = len(neighbors2)
                    index_dis = (dist + dist_inv.t()).flatten().argsort(descending=False)
                    index_sim = (candi_probs + candi_probs_inv.t()).flatten().argsort(descending=True)

                    index_dis_e1 = index_dis // num2
                    index_dis_e2 = index_dis % num2
                    index_sim_e1 = index_sim // num2
                    index_sim_e2 = index_sim % num2
                    aligned_dis_e1 = torch.full((num1,), -1)
                    aligned_dis_e2 = torch.full((num2,), -1)
                    aligned_sim_e1 = torch.full((num1,), -1)
                    aligned_sim_e2 = torch.full((num2,), -1)

                    for _ in range(num1*num2):
                        if aligned_dis_e1[index_dis_e1[_]] >= 0 or aligned_dis_e2[index_dis_e2[_]] >= 0:
                            continue
                        aligned_dis_e1[index_dis_e1[_]] = index_dis_e2[_]
                        aligned_dis_e2[index_dis_e2[_]] = index_dis_e1[_]
                    for _ in range(num1*num2):
                        if aligned_sim_e1[index_sim_e1[_]] >= 0 or aligned_sim_e2[index_sim_e2[_]] >= 0:
                            continue
                        aligned_sim_e1[index_sim_e1[_]] = index_sim_e2[_]
                        aligned_sim_e2[index_sim_e2[_]] = index_sim_e1[_]
                    for i in range(num1):
                        if aligned_dis_e1[i] == aligned_sim_e1[i]:
                            prob = candi_probs[i][aligned_dis_e1[i]]
                            prob_inv = candi_probs_inv[aligned_dis_e1[i]][i]
                            d = dist[i][aligned_dis_e1[i]]
                            d_inv = dist_inv[aligned_dis_e1[i]][i]
                            if neighbors1[i] not in new_alignment_dict1 and neighbors2[aligned_dis_e1[i]] not in new_alignment_dict2:
                                new_alignment_dict1[neighbors1[i]] = prob
                                new_alignment_dict2[neighbors2[aligned_dis_e1[i]]] = prob_inv
                            elif neighbors1[i] in new_alignment_dict1:
                                if new_alignment_dict1[neighbors1[i]] <= prob:
                                    value_to_remove = neighbors1[i]  # 要移除的值
                                    new_alignment = [row for row in new_alignment if row[0] != value_to_remove]
                                else:
                                    continue
                            elif neighbors2[aligned_dis_e1[i]] in new_alignment_dict2:
                                if new_alignment_dict2[neighbors2[aligned_dis_e1[i]]] <= prob_inv:
                                    value_to_remove = neighbors2[aligned_dis_e1[i]]  # 要移除的值
                                    new_alignment = [row for row in new_alignment if row[1] != value_to_remove]
                                else:
                                    continue
                            else:
                                continue
                            new_alignment.append([neighbors1[i], neighbors2[aligned_dis_e1[i]]])
        if data.new_set.shape[0] > 0:
            data.new_set = intersect_nx2(data.new_set.cpu(), torch.tensor(new_alignment)).to(data.train_set.device)
        else:
            data.new_set = torch.tensor(new_alignment, device=data.train_set.device)
        for i in data.train_set.cpu().numpy():
            new_alignment.append(i)

    p_set = torch.tensor(new_alignment, device=data.train_set.device)

    for ent1, ent2 in data.train_set.cpu().numpy():
        if ent1 == ent2:
            ptrue += 1
        else:
            pfalse += 1
    rec_num = ptrue + pfalse
    if rec_num == 0:
        pacc = 0
    else:
        pacc = ptrue / (ptrue + pfalse)
    print("new labels num:", ptrue + pfalse)
    print("joint distr threshold & mutual nearest acc", pacc)


def generate_labels(x1, x2, data):
    pair = data.test_set
    device = pair.device

    all_entities1 = torch.arange(x1.shape[0]).to(device)
    all_entities2 = torch.arange(x2.shape[0]).to(device)

    other_entities1 = generate_other_entities(all_entities1, data.train_set[:, 0])
    other_entities2 = generate_other_entities(all_entities2, data.train_set[:, 1])
    print("other_entities1.shape, other_entities2.shape:", other_entities1.shape, other_entities2.shape)

    num1 = other_entities1.shape[0]
    num2 = other_entities2.shape[0]

    # pair_num = pair.size(0)
    # S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    S = -torch.cdist(x1[other_entities1], x2[other_entities2], p=1).cpu()
    # index = S.flatten().argsort(descending=True)
    index = (S.softmax(1) + S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index // num2
    index_e2 = index % num2
    aligned_e1 = torch.zeros(num1, dtype=torch.bool)
    aligned_e2 = torch.zeros(num2, dtype=torch.bool)
    true_aligned = 0
    num = 0
    for _ in range(num1*5):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if other_entities1[index_e1[_]] == other_entities2[index_e2[_]]:
            true_aligned += 1
            # print(other_entities1[index_e1[_]], other_entities2[index_e2[_]])
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
        # data.train_set.append([index_e1[_], index_e2[_]])
        new_element = torch.tensor([other_entities1[index_e1[_]], other_entities2[index_e2[_]]])
        num += 1

        # 使用torch.cat将新元素添加到data.train_set
        data.train_set = torch.cat((data.train_set, new_element.unsqueeze(0).to(device)), dim=0)
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned / num * 100))



