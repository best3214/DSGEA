# -*- coding: utf-8 -*-
import numpy as np
import torch

import torch.nn.functional as F
from tqdm import tqdm

from paris.indicator_paris import apply_reasoning_torch, apply_reasoning_torch1
from utils import generate_other_entities

def _set_neural_prob_mtx(neural_prob_mtx: torch.Tensor, labelled_entities):
    ent2_num = neural_prob_mtx.size()[1]
    print("_set_neural_prob_mtx neural_prob_mtx.deice, 判断是否为cuda:", neural_prob_mtx.device)
    for ent1, ent2 in labelled_entities:
        onehot = torch.zeros(size=(ent2_num,), device=neural_prob_mtx.device)
        onehot[ent2] = 1
        neural_prob_mtx[ent1] = onehot
    candi_sum_prob, candi_probs, candidates, neural_prob_mtx = _generate_candidates(neural_prob_mtx)

    return candi_sum_prob, candi_probs, candidates, neural_prob_mtx

def set_stuff_about_neu_model(prob_mtx: torch.Tensor, labelled_entities):
    candi_sum_prob, candi_probs, candidates, neural_prob_mtx = _set_neural_prob_mtx(prob_mtx, labelled_entities)
    return candi_sum_prob, candi_probs, candidates, neural_prob_mtx

def _generate_candidates(neural_prob_mtx, topK=10):
    candi_probs, candi_idxes = torch.topk(neural_prob_mtx, dim=1, k=topK)
    candi_sum_prob = torch.sum(candi_probs, dim=1, keepdim=False)
    # candidates = candi_idxes
    return candi_sum_prob, candi_probs, candi_idxes, neural_prob_mtx

def compute_features(candidates, neural_prob_mtx, data, device, inv=False):
    ent_arr = torch.arange(0, len(candidates), device=device)
    if inv:
        features1 = apply_reasoning_torch(ent_arr, candidates, neural_prob_mtx, data, device, inv)

    else:
        features1 = apply_reasoning_torch(ent_arr, candidates, neural_prob_mtx, data, device)
    features = features1.unsqueeze(dim=2)
    # features2 = paris_model.apply_negative_reasoning_rule(ent_arr, self.candidates, self.neural_prob_mtx)
    # features = features2.unsqueeze(dim=2)
    # features = torch.stack([features1, features2], dim=-1)
    # features_cache = features
    return features

def compute_features1(neural_prob_mtx, data, most_sim_pair, device):
    # ent_arr = torch.arange(0, len(candidates), device=device)
    print("compute_features1--device:", device)
    second_device = torch.device('cpu')
    # def apply_reasoning_torch1(neighbors1: np.array, neighbors2: np.array, prob_mtx: torch.Tensor, data, device,
    #                            ratio=args.joint_distr_thr):
    all_entities1 = torch.arange(neural_prob_mtx.shape[0]).to(second_device)
    all_entities2 = torch.arange(neural_prob_mtx.shape[1]).to(second_device)
    neighbors1 = []
    neighbors2 = []
    print("all_entities1.shape, all_entities2.shape", all_entities1.shape, all_entities2.shape)

    for ent1, ent2 in tqdm(most_sim_pair.numpy(), ascii=True, desc="Generating neighbors"):
        # print("ent1, ent2:", ent1, ent2)
        if ent1 in data.neigh1 and ent2 in data.neigh2:
            neighbors1.append(data.neigh1[ent1])
            neighbors2.append(data.neigh2[ent2])
    assert len(neighbors1) == len(neighbors2)
    neighbors1 = np.array(neighbors1)
    neighbors2 = np.array(neighbors2)


    other_entities1 = generate_other_entities(all_entities1, data.temp_set[:, 0])
    other_entities2 = generate_other_entities(all_entities2, data.temp_set[:, 1])
    # if inv:
    #     features = apply_reasoning_torch1(other_entities1, other_entities2, neural_prob_mtx, data, device, inv)
    #
    # else:
    apply_reasoning_torch1(neighbors1, neighbors2, other_entities1, other_entities2, neural_prob_mtx, data, device)
    # features = features1.unsqueeze(dim=2)
    # features2 = paris_model.apply_negative_reasoning_rule(ent_arr, self.candidates, self.neural_prob_mtx)
    # features = features2.unsqueeze(dim=2)
    # features = torch.stack([features1, features2], dim=-1)
    # features_cache = features


def coordinate_ascend(candidates, neural_prob_mtx, candi_sum_prob, candi_probs, data, device, inv=False):
    print("coordinate_ascend--neural_prob_mtx:", neural_prob_mtx[:10])
    with torch.no_grad():
        # if self.features_cache is None:
        #     features = self.compute_features()
        # else:
        #     features = self.features_cache
        if not inv:
            features = compute_features(candidates, neural_prob_mtx, data, device)
        else:
            features = compute_features(candidates, neural_prob_mtx, data, device, inv)

        candi_prob_mtx = conditional_p(features).to(torch.device("cpu"))
        fe = candi_prob_mtx.cpu().detach().numpy()

        # 打开文件并写入数据

        # hybrid_prob_mtx = neural_prob_mtx
        # mtx = torch.zeros_like(neural_prob_mtx)
        # mtx.scatter_(dim=1, index=candidates,
        #                          src=candi_prob_mtx)
        neural_prob_mtx.scatter_(dim=1, index=candidates,
                                 src=candi_prob_mtx*candi_sum_prob.unsqueeze(dim=1))
        with open('output_improve_probs.txt', 'w', encoding='utf-8') as f:
            for row in neural_prob_mtx.cpu().numpy()[:1000]:
                # 将每行的元素转换为字符串并用空格分隔，然后写入文件
                f.write(' '.join(map(str, row)) + '\n')
        # neural_prob_mtx = neural_prob_mtx * candi_prob_mtx
        # np.savez(os.path.join(self.conf.output_dir, "coordinate_ascent_prob_mtx.npz"),  improved_prob_mtx=hybrid_prob_mtx.cpu().numpy())
    # return mtx
    return neural_prob_mtx
def coordinate_ascend1(neural_prob_mtx, data, most_sim_pair, device):
    with torch.no_grad():
        # if self.features_cache is None:
        #     features = self.compute_features()
        # else:
        #     features = self.features_cache
        # if not inv:
        compute_features1(neural_prob_mtx, data, most_sim_pair, device)
        # else:
        #     features = compute_features1(other_entities1, other_entities2, neural_prob_mtx, data, device, inv)

        # candi_prob_mtx = conditional_p(features).to(torch.device("cpu"))
        # fe = candi_prob_mtx.cpu().detach().numpy()

        # 打开文件并写入数据

        # hybrid_prob_mtx = neural_prob_mtx
        # mtx = torch.zeros_like(neural_prob_mtx)
        # mtx.scatter_(dim=1, index=candidates,
        #                          src=candi_prob_mtx)
        # neural_prob_mtx.scatter_(dim=1, index=candidates,
        #                          src=candi_prob_mtx*candi_sum_prob.unsqueeze(dim=1))
        # with open('output_improve_probs.txt', 'w', encoding='utf-8') as f:
        #     for row in neural_prob_mtx.cpu().numpy()[:1000]:
                # 将每行的元素转换为字符串并用空格分隔，然后写入文件
    #             f.write(' '.join(map(str, row)) + '\n')
    #     # neural_prob_mtx = neural_prob_mtx * candi_prob_mtx
    #     # np.savez(os.path.join(self.conf.output_dir, "coordinate_ascent_prob_mtx.npz"),  improved_prob_mtx=hybrid_prob_mtx.cpu().numpy())
    # # return mtx
    # return neural_prob_mtx

def conditional_p(features: torch.Tensor):
    candi_score_mtx = local_func(features)
    factor_prob_mtx = F.normalize(candi_score_mtx, dim=1, p=1)
    # print("conditional_p--factor_prob_mtx:", factor_prob_mtx)
    return factor_prob_mtx

def local_func(features: torch.Tensor):
    features = features.to(dtype=torch.float32, device=torch.device("cpu"))
    # candi_score_mtx = torch.exp(self.linear_layer(features))
    candi_score_mtx = torch.exp(features)
    fea_shape = candi_score_mtx.shape
    candi_score_mtx = candi_score_mtx.reshape(shape=fea_shape[:2])
    return candi_score_mtx