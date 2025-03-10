# import torch
#
# from utils import generate_topk_sim_dict
# import numpy as np
#
# # 假设 dist 是一个一维张量
# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=torch.float32)
# b = torch.tensor([[1, 2, 4], [1, 5, 6], [7, 8, 0]], dtype=torch.float32)
# o1 = torch.tensor([1, 2])
# sim_mtx_inv = a.index_select(1, o1).t()
# o2 = torch.tensor([0, 1, 2])
# d = [1,2,3]
# print(sim_mtx_inv)
#
# # c = generate_topk_sim_dict(a, b, o1, o2, 2)
# # print(c)
#
# print(intersect_nx2(a,b))
import torch

assoc = torch.full((3,), -1, dtype=torch.long)
print(assoc)