import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree

from data import generate_class
from paris.CSLS_torch import Evaluator
import numpy as np
from classifier import Classifier

from paris.components_base import convert_simi_to_probs7
from conf import parse_args

args = parse_args()

class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        # print(edge_index_i, x.size(0))
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

    
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, c_hidden, r_hidden):
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h3 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h4 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t3 = nn.Linear(c_hidden, 1, bias=False)

        self.a_r1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_r2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_c = nn.Linear(c_hidden, 1, bias=False)

        self.w_h = nn.Linear(e_hidden, c_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, c_hidden, bias=False)

        self.highway = Highway(c_hidden)



    def forward(self, x_e, edge_index, rel, triple_num, r_emb, class_index, head_class):
        edge_index_h, edge_index_t = edge_index
        # print(1)
        x_r_h = F.relu(self.w_h(x_e))
        x_r_t = F.relu(self.w_t(x_e))

        # （1）拆分三元组
        # 计算拆分系数alpha
        e1 = (self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_t1(x_r_t).squeeze()[edge_index_t]) / 2
        e_r = self.a_r1(r_emb).squeeze()[rel]
        e1 = e1 + e_r
        alpha = softmax(F.leaky_relu(e1).float(), triple_num)

        # （2）按照类别聚合
        # 2.1计算尾实体聚合度e_t
        x_t = x_r_t[edge_index_t]
        x_t = x_t * alpha.view(-1, 1)
        x_t = F.normalize(x_t, dim=1, p=2).requires_grad_()
        e_t = self.a_t2(x_t).squeeze()

        # 2.2计算头实体聚合度e_h
        e_h = self.a_h2(x_r_h).squeeze()[edge_index_h]

        # 2.3计算关系聚合度e_r
        e_r = self.a_r2(r_emb).squeeze()[rel]
        e_r_e_h = self.a_h3(x_r_h).squeeze()[edge_index_h]
        e_r_e_t = self.a_t3(x_t).squeeze()
        e_r = (e_r + (e_r_e_h + e_r_e_t) / 2) / 2

        # e = self.a_h3(x_r_h).squeeze()[edge_index_h] + self.a_h4(x_r_t).squeeze()[edge_index_t]
        # e2_c = self.a_t3(x_r_h).squeeze()[edge_index_h] + self.a_t4(x_r_t).squeeze()[edge_index_t]
        # print(4)

        # e_r_e = alpha * e2
        # # e2_c = alpha * e2_c
        # # print(5)
        #
        # e_r = e_r_e + e_r
        # e_t = e_t * alpha

        # 2.4计算类别划分系数beta
        e = e_t + e_r + e_h
        # print("edge_index.size(), e.size(), class_index.size():", edge_index.size(), e.size(), class_index.size())
        beta = softmax(F.leaky_relu(e).float(), class_index)

        # 2.5计算类别嵌入x_class
        x_class = spmm(torch.cat([class_index.view(1, -1), edge_index_t.view(1, -1)], dim=0), beta, class_index.max() + 1, x_t.size(0),
                     x_t)

        # （3）聚合类嵌入
        e_c = self.a_c(x_class).squeeze()
        e_h_c = self.a_h4(x_r_h).squeeze()[head_class]
        e_c = e_c + e_h_c

        gama = softmax(F.leaky_relu(e_c).float(), head_class)
        class_num = torch.arange(class_index.max() + 1).to(x_t.device)
        # print("head_class.size(), class_index.size(), gama.size(), x_class.size():",
        #       head_class.size(), class_index.size(), gama.size(), x_class.size())
        x_e_h = spmm(torch.cat([head_class.view(1, -1), class_num.view(1, -1)], dim=0), gama, x_e.size(0), class_index.max() + 1,
                     x_class)
        x_e_h = self.highway(x_r_h, x_e_h)

        return x_e_h

class GAT_E_T(nn.Module):
    def __init__(self, e_hidden, c_hidden, r_hidden):
        super(GAT_E_T, self).__init__()
        self.a_h1 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h3 = nn.Linear(c_hidden, 1, bias=False)
        self.a_h4 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(c_hidden, 1, bias=False)
        self.a_t3 = nn.Linear(c_hidden, 1, bias=False)

        self.a_r1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_r2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_c = nn.Linear(c_hidden, 1, bias=False)

        self.w_h = nn.Linear(e_hidden, c_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, c_hidden, bias=False)

        self.highway = Highway(c_hidden)



    def forward(self, x_e, edge_index, rel, triple_num, r_emb, class_index, tail_class):
        edge_index_h, edge_index_t = edge_index
        x_r_h = F.relu(self.w_h(x_e))
        x_r_t = F.relu(self.w_t(x_e))

        # （1）拆分三元组
        # 计算拆分系数alpha
        e1 = (self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_t1(x_r_t).squeeze()[edge_index_t]) / 2
        e_r = self.a_r1(r_emb).squeeze()[rel]
        e1 = e1 + e_r
        alpha = softmax(F.leaky_relu(e1).float(), triple_num)

        # （2）按照类别聚合
        # 2.1计算头实体聚合度e_h
        x_h = x_r_h[edge_index_h]
        x_h = x_h * alpha.view(-1, 1)
        x_h = F.normalize(x_h, dim=1, p=2).requires_grad_()
        e_h = self.a_h2(x_h).squeeze()

        # 2.2计算尾实体聚合度e_t
        e_t = self.a_t2(x_r_t).squeeze()[edge_index_t]

        # 2.3计算关系聚合度e_r
        e_r = self.a_r2(r_emb).squeeze()[rel]
        e_r_e_t = self.a_t3(x_r_t).squeeze()[edge_index_t]
        e_r_e_h = self.a_h3(x_h).squeeze()
        e_r = (e_r + (e_r_e_h + e_r_e_t) / 2) / 2

        # 2.4计算类别划分系数beta
        e = e_t + e_r + e_h
        # print("edge_index.size(), e.size(), class_index.size():", edge_index.size(), e.size(), class_index.size())
        beta = softmax(F.leaky_relu(e).float(), class_index)

        # 2.5计算类别嵌入x_class
        x_class = spmm(torch.cat([class_index.view(1, -1), edge_index_h.view(1, -1)], dim=0), beta, class_index.max() + 1, x_h.size(0),
                     x_h)

        # e1 = self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_h2(x_r_t).squeeze()[edge_index_t]
        # e2 = self.a_t1(x_r_h).squeeze()[edge_index_h] + self.a_t2(x_r_t).squeeze()[edge_index_t]
        # e_t = self.a_t3(x_r_t).squeeze()[edge_index_t]
        # e_h = self.a_h3(x_r_h).squeeze()[edge_index_h]
        #
        # alpha = softmax(F.leaky_relu(e1).float(), triple_num)
        #
        # # e = self.a_h3(x_r_h).squeeze()[edge_index_h] + self.a_h4(x_r_t).squeeze()[edge_index_t]
        # # e2_c = self.a_t3(x_r_h).squeeze()[edge_index_h] + self.a_t4(x_r_t).squeeze()[edge_index_t]
        #
        # e_r_e = alpha * e2
        # # e2_c = alpha * e2_c
        #
        # e_r = self.a_r(r_emb).squeeze()[rel]
        # e_r = e_r_e + e_r
        # e_h = e_h * alpha
        # # e_t = e_t * alpha
        # e = e_t + e_r + e_h
        # beta = softmax(F.leaky_relu(e).float(), class_index)
        #
        # x_h = x_r_h[edge_index_h]
        #
        # x_class = spmm(torch.cat([class_index.view(1, -1), edge_index_h.view(1, -1)], dim=0), beta, class_index.max() + 1, x_h.size(0),
        #              x_h)

        # （3）聚合类嵌入
        e_c = self.a_c(x_class).squeeze()
        e_t_c = self.a_h4(x_r_t).squeeze()[tail_class]
        e_c = e_c + e_t_c

        gama = softmax(F.leaky_relu(e_c).float(), tail_class)
        class_num = torch.arange(class_index.max() + 1).to(x_e.device)

        x_e_t = spmm(torch.cat([tail_class.view(1, -1), class_num.view(1, -1)], dim=0), gama, x_e.size(0), class_index.max() + 1,
                     x_class)
        x_e_t = self.highway(x_r_t, x_e_t)

        return x_e_t

# class GAT_R_to_E(nn.Module):
#     def __init__(self, e_hidden, r_hidden):
#         super(GAT_R_to_E, self).__init__()
#         self.a_h = nn.Linear(e_hidden, 1, bias=False)
#         self.a_t = nn.Linear(e_hidden, 1, bias=False)
#         self.a_r = nn.Linear(r_hidden, 1, bias=False)
#
#     def forward(self, x_e, x_r, edge_index, rel):
#         edge_index_h, edge_index_t = edge_index
#         e_h = self.a_h(x_e).squeeze()[edge_index_h]
#         e_t = self.a_t(x_e).squeeze()[edge_index_t]
#         e_r = self.a_r(x_r).squeeze()[rel]
#         alpha = softmax(F.leaky_relu(e_h+e_r).float(), edge_index_h)
#         x_e_h = spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
#         alpha = softmax(F.leaky_relu(e_t+e_r).float(), edge_index_t)
#         x_e_t = spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
#         x = torch.cat([x_e_h, x_e_t], dim=1)
#         return x
    

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
        
    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x
    
    
class DSGEA(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100, c_hidden=150, r_num=None, c_num=150):
        super(DSGEA, self).__init__()
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.gcn3 = GCN(e_hidden)
        self.gcn4 = GCN(e_hidden)
        self.gcn5 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.highway3 = Highway(e_hidden)
        self.highway4 = Highway(e_hidden)

        self.gat_e_to_r = GAT_E_to_R(e_hidden, c_hidden, r_hidden)
        self.gat_e_T = GAT_E_T(e_hidden, c_hidden, r_hidden)

        # self.gat_r_to_e = GAT_R_to_E(e_hidden, r_hidden)
        self.gat = GAT(e_hidden * 2)
        self.gat1 = GAT(e_hidden)
        self.classifier = Classifier(r_hidden, int(r_hidden / 2), c_num)
        if r_num is not None:
            self.r_emb = nn.Embedding(r_num, r_hidden)

    def forward(self, x_e, edge_index, rel, edge_index_all, *args):
        # print(x_e.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(args) == 1:
            start_num = args[0]

            r_emb = F.normalize(self.r_emb(torch.arange(start_num, rel.max()+1+start_num).to(device)), p=2, dim=1)
            r_class = self.classifier(r_emb)
            self.r_class_id = r_class[rel]
            triple_num, class_index_head, head_class, class_index_tail, tail_class = generate_class(edge_index.cpu(), self.r_class_id.cpu())

            triple_num, class_index_head, head_class, class_index_tail, tail_class = triple_num.to(device), class_index_head.to(device), head_class.to(device), class_index_tail.to(device), tail_class.to(device)
            x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
            x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
            x_e_h = self.gat_e_to_r(x_e, edge_index, rel, triple_num, r_emb, class_index_head, head_class)
            # x_e_h = self.gcn3(x_e_h, edge_index_all)
            x_e_h_t = torch.cat([x_e_h, self.gat_e_T(x_e, edge_index, rel, triple_num, r_emb, class_index_tail, tail_class)], dim=1)

            x_e1 = torch.cat([x_e, x_e_h_t], dim=1)
            x_e = torch.cat([x_e, self.gat(x_e1, edge_index_all)], dim=1)
        else:
            triple_num, class_index_head, head_class, class_index_tail, tail_class, start_num = args
            r_emb = F.normalize(self.r_emb(torch.arange(start_num, rel.max() + 1 + start_num).to(device)), p=2, dim=1)
            r_class = self.classifier(r_emb)
            self.r_class_id = r_class[rel]

            x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
            x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
            x_e_h = self.gat_e_to_r(x_e, edge_index, rel, triple_num, r_emb, class_index_head, head_class)
            # x_e_h = self.gcn3(x_e_h, edge_index_all)
            x_e_h_t = torch.cat(
                [x_e_h, self.gat_e_T(x_e, edge_index, rel, triple_num, r_emb, class_index_tail, tail_class)], dim=1)

            x_e1 = torch.cat([x_e, x_e_h_t], dim=1)
            x_e = torch.cat([x_e, self.gat(x_e1, edge_index_all)], dim=1)

        return x_e
    # def forward(self, x_e, edge_index, rel, edge_index_all, triple_num, class_index_head, head_class,class_index_tail, tail_class, start_num=0):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     r_emb = F.normalize(self.r_emb(torch.arange(start_num, rel.max() + 1 + start_num).to(device)), p=2, dim=1)
    #     r_class = self.classifier(r_emb)
    #     self.r_class_id = r_class[rel]
    #     # triple_num, class_index_head, head_class, class_index_tail, tail_class = generate_class(edge_index.cpu(),
    #     #                                                                                         self.r_class_id.cpu())
    #
    #     # triple_num, class_index_head, head_class, class_index_tail, tail_class = triple_num.to(
    #     #     device), class_index_head.to(device), head_class.to(device), class_index_tail.to(device), tail_class.to(
    #     #     device)
    #     x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
    #     x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
    #     x_e_h = self.gat_e_to_r(x_e, edge_index, rel, triple_num, r_emb, class_index_head, head_class)
    #     # x_e_h = self.gcn3(x_e_h, edge_index_all)
    #     x_e_h_t = torch.cat(
    #         [x_e_h, self.gat_e_T(x_e, edge_index, rel, triple_num, r_emb, class_index_tail, tail_class)], dim=1)
    #
    #     x_e1 = torch.cat([x_e, x_e_h_t], dim=1)
    #     x_e = torch.cat([x_e, self.gat(x_e1, edge_index_all)], dim=1)
    #     return x_e

    def classify(self, x_r):
        r_class_id = self.classifier(x_r)
        return r_class_id




    # def PrX(self, list1, list2, neigh, r_dict, emb, fr, min_sim, device):
    #     emb = self.linear(emb)
    #     print("emb.device", emb.device)
    #     print("list1.size(), list2.size()", len(list1), len(list2))
    #     print("开始邻域传播相似度...")
    #
    #     rows = []
    #     cols = []
    #     neigh_list1 = []
    #     neigh_list2 = []
    #     row_num = 0
    #     col_num = 0
    #     fr1 = []
    #     fr2 = []
    #     # print("list1:",len(list1), list1)
    #     # print(neigh)
    #     for num in range(len(list1)):
    #         e1, e2 = list1[num], list2[num]
    #         if e1 in neigh and e2 in neigh:
    #             for neighbor in neigh[e1]:
    #                 neigh_list1.append(neighbor)
    #                 # prr = 1
    #                 temp_list1 = []
    #                 for r in r_dict[(e1, neighbor)]:
    #                     # prr = prr * fr[r]
    #                     temp_list1.append(fr[r])
    #                 # fr1.append(prr)
    #                 fr1.append(temp_list1)
    #                 row_num += 1
    #             rows.append(row_num)
    #
    #             # for e2 in list2:
    #             for neighbor in neigh[e2]:
    #                 neigh_list2.append(neighbor)
    #                 # prr = 1
    #                 temp_list2 = []
    #                 for r in r_dict[(e2, neighbor)]:
    #                     # prr = prr * fr[r]
    #                     temp_list2.append(fr[r])
    #                 col_num += 1
    #                 fr2.append(temp_list2)
    #             cols.append(col_num)
    #     print("标签数据的邻居列表准备完毕！")
    #     print(len(neigh_list1), len(neigh_list2))
    #
    #     # 设置掩膜
    #     mask = torch.zeros(len(neigh_list1), len(neigh_list2), device=device)
    #     start_row = 0
    #     start_col = 0
    #     for i in range(len(rows)):
    #         end_row = rows[i]
    #         end_col = cols[i]
    #         mask[start_row:end_row, start_col:end_col] = 1
    #         start_row = end_row
    #         start_col = end_col
    #     fr1 = torch.tensor([x + [0] * (15 - len(x)) for x in fr1], device=device)
    #     fr2 = torch.tensor([x + [0] * (15 - len(x)) for x in fr2], device=device)
    #     # fr1 = torch.tensor(fr1)
    #     # fr2 = torch.tensor(fr2)
    #     # neigh_fr = torch.mul(fr1.repeat(len(neigh_list2)).view(len(neigh_list2), -1).t(), fr2.repeat(len(neigh_list1)).view(len(neigh_list1), -1))
    #     # neigh_fr = torch.where(mask > 0, neigh_fr, 0)
    #
    #     # emb1 = torch.stack([emb[n1] for n1 in neigh_list1]).unsqueeze(0).unsqueeze(0)
    #     # emb2 = torch.stack([emb[n2] for n2 in neigh_list2]).unsqueeze(0).unsqueeze(0)
    #     emb1 = torch.stack([emb[n1] for n1 in neigh_list1])
    #     emb2 = torch.stack([emb[n2] for n2 in neigh_list2])
    #     print("emb1.device:", emb1.device)
    #     print(len(emb[0]), len(emb1), emb1.size(), emb2.size())
    #     # conv_layer1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, len(emb[0])), stride=1)
    #     # conv_layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, len(emb[0])), stride=1)
    #     # print(self.conv_layer1(emb1).size())
    #     # output1 = self.linear1(emb1).view(-1)
    #     # output2 = self.linear2(emb2).view(-1)
    #     # print(output2.size(), output1.size())
    #     # output = output1.repeat(len(neigh_list2)).view(len(neigh_list2), -1).t() + output2.repeat(
    #     #     len(neigh_list1)).view(len(neigh_list1), -1)
    #     # 假设 self.linear1 和 self.linear2 是已经定义的线性层
    #     # emb1 和 emb2 是输入的嵌入向量
    #     output1 = self.linear1(emb1).squeeze()  # 假设 emb1 已经是正确的形状
    #     output2 = self.linear2(emb2).squeeze()  # 假设 emb2 已经是正确的形状
    #
    #     # 使用 unsqueeze 增加一个维度
    #     output1 = output1.unsqueeze(0).expand(len(neigh_list2), -1)
    #     output2 = output2.unsqueeze(0).expand(len(neigh_list1), -1)
    #
    #     # 转置和相加
    #     output = output1.t() + output2
    #
    #     print(output.dtype, mask.dtype)
    #     # output.to(torch.float32)
    #     # mask.to(torch.float32)
    #     inf_tensor = torch.tensor(float('-inf'), dtype=torch.float16, device=device)
    #     output = torch.where(mask > 0, output, inf_tensor)
    #     logits = F.softmax(output, dim=1)
    #     # sim_matrix = torch.zeros(len(neigh_list1), len(neigh_list2))
    #
    #     fr1 = fr1.unsqueeze(1)
    #     print("打印fr1值", fr1.size())
    #
    #     sim_left = fr1 * logits.unsqueeze(2)
    #     # one_matrix = torch.ones_like(sim_left)
    #     sim_left = 1 - sim_left
    #     sim_left_matrix = torch.prod(sim_left, dim=2)
    #     print(sim_left_matrix.size())
    #
    #     fr2 = fr2.unsqueeze(0)
    #     sim_right = fr2 * logits.unsqueeze(2)
    #     sim_right = 1 - sim_right
    #     sim_right_matrix = torch.prod(sim_right, dim=2)
    #     print(sim_right_matrix.size())
    #     sim_matrix = 1 - sim_left_matrix * sim_right_matrix
    #     print(sim_matrix.dtype)
    #     sim_matrix = torch.where(mask > 0, sim_matrix, torch.tensor(0, dtype=torch.float32, device=device))
    #
    #
    #     sim_max_values, sim_max_indices = torch.max(sim_matrix, dim=1)
    #     # print(sim_max_values[0:50], sim_max_indices[0:50], rows[0:10])
    #
    #     start = 0
    #     new_ILL = []
    #     # 筛选相似度最高的实体对
    #     for i in range(len(cols)):
    #         end = rows[i]
    #         matrix_temp = sim_max_values[start:end]
    #         # print(i, matrix_temp)
    #         max_value_left, max_indice_left = torch.max(matrix_temp, dim=0)
    #         # 相似度超过阈值
    #         if max_value_left > min_sim:
    #             max_value_right, max_indice_right = torch.max(sim_matrix[:, sim_max_indices[max_indice_left + start]],
    #                                                           dim=0)
    #             # print(max_indice_right, start, max_indice_left)
    #             if max_indice_right == start + max_indice_left:
    #                 new_ILL.append([neigh_list1[max_indice_right], neigh_list2[sim_max_indices[max_indice_right]]-19388])
    #         start = end
    #     print("本轮筛选出的对齐实体对个数为：", len(new_ILL), new_ILL)
    #     return torch.tensor(new_ILL).to(device)

    def predict_simi(self, ent1_embs, ent2_embs, device):
        # emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        # embs = emb_res["embs"]
        # ent1_ids = emb_res["ent1_ids"]
        # ent2_ids = emb_res["ent2_ids"]
        # ent1_embs = x1
        # ent2_embs = x2

        # simi_mtx = sim_handler(ent1_embs, ent2_embs, k=10, nums_threads=100)

        evaluator = Evaluator(device=device)
        # simi_mtx, simi_mtx_inv = evaluator.dis_sim(ent1_embs, ent2_embs)
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs)

        return simi_mtx

    def predict(self, x1, x2, device, no_sim2prob=False):
        simi_mtx = self.predict_simi(x1, x2, device)
        # simi_mtx_inv = self.predict_simi(x2, x1, device)
        # simi_mtx = torch.tensor(simi_mtx, device=torch.device("cpu"))
        # simi_mtx_inv = simi_mtx.t()


        if not no_sim2prob:
            prob_mtx = convert_simi_to_probs7(simi_mtx, device, args.output)
            prob_mtx_inv = convert_simi_to_probs7(simi_mtx, device, args.output, inv=True)
            prob_mtx = torch.tensor(prob_mtx, device=torch.device("cpu"))
            prob_mtx_inv = torch.tensor(prob_mtx_inv, device=torch.device("cpu"))
        else:
            prob_mtx = simi_mtx
            ma = np.amax(prob_mtx, axis=1)
            mi = np.amin(prob_mtx, axis=1)
            prob_mtx = prob_mtx - mi.reshape(-1, 1) / (ma.reshape(-1, 1) - mi.reshape(-1, 1))
            prob_mtx_inv = simi_mtx.transpose()
            ma = np.amax(prob_mtx_inv, axis=1)
            mi = np.amin(prob_mtx_inv, axis=1)
            prob_mtx_inv = prob_mtx_inv - mi.reshape(-1, 1) / (ma.reshape(-1, 1) - mi.reshape(-1, 1))
            prob_mtx = torch.tensor(prob_mtx, device=torch.device(device))
            prob_mtx_inv = torch.tensor(prob_mtx_inv, device=torch.device(device))

        del simi_mtx
        return prob_mtx, prob_mtx_inv
        # return torch.tensor(simi_mtx), torch.tensor(simi_mtx_inv)


    # def train_model_with_observed_n_latent_labels(self, ite):
    #
    #     # if self.conf.py_exe_fn is None:
    #     #     runner = Runner(self.conf.data_dir, self.conf.output_dir, enhanced=True,
    #     #                     max_train_epoch=self.conf.max_train_epoch,
    #     #                     max_continue_epoch=self.conf.max_continue_epoch,
    #     #                     eval_freq=self.conf.eval_freq,
    #     #                     )
    #     #     runner.restore_model(self.conf.restore_from_dir)
    #     #     kg1id, kg2id = self.conf.data_name.split("_")
    #         # runner.train()
    #     if args.continue_training in ("supervised", "sup"):
    #         self.continue_training()
    #     # elif args.continue_training in ("iterative", "semi"):
    #     #     runner.continue_iterative_training()
    #     else:
    #         raise Exception("unknown initial training method")
    #     # runner.save(save_metrics=self.conf.neu_save_metrics)
    #     # else:
    #     #     cmd_fn = self.conf.py_exe_fn
    #     #     cur_dir = os.path.dirname(os.path.realpath(__file__))
    #     #     script_fn = os.path.join(cur_dir, "RREA/runner.py")
    #     #     args_str = f"--data_dir={self.conf.data_dir} --output_dir={self.conf.output_dir} " \
    #     #                f"--max_train_epoch={self.conf.max_train_epoch} --max_continue_epoch={self.conf.max_continue_epoch} " \
    #     #                f"--initial_training={self.conf.initial_training} --continue_training={self.conf.continue_training} " \
    #     #                f"--enhanced=True --restore_from_dir={self.conf.restore_from_dir}"
    #     #     env = os.environ.copy()
    #     #     print(cmd_fn + " " + script_fn + " " + args_str)
    #     #     env["CUDA_VISIBLE_DEVICES"] = self.conf.tf_device
    #     #     ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
    #     #     if ret.returncode != 0:
    #     #         raise Exception("RREA did not run successfully")
    #
    #     # train simi to prob model
    #     simi_mtx = self.predict_simi()
    #     print('simi_mtx.shape', simi_mtx.shape)
    #     simi2prob_model = SimiToProbModule(conf=self.conf)
    #     simi2prob_model.restore_from(self.conf.restore_from_dir)
    #     simi2prob_model.train_model(simi_mtx)
    #     simi2prob_model_inv = SimiToProbModule(conf=self.conf, inv=True)
    #     simi2prob_model_inv.restore_from(self.conf.restore_from_dir)
    #     simi2prob_model_inv.train_model(simi_mtx)

    # def continue_training(self, data):
    #     epoch = 0
        # log_fn = os.path.join(self.output_dir, "training_log.json")
        # if os.path.exists(log_fn):
        #     with open(log_fn) as file:
        #         logs = json.loads(file.read())
        # else:
        #     logs = {}
        # no_improve_num = 0
        # if os.path.exists(os.path.join(self.output_dir, "model.ckpt")):
        #     self.restore_model()
        # while True:
        # for _ in trange(args.max_continue_epoch, desc="continue training RREA"):
        #     train_set = data.train_set.t()
        #     inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
        #     inputs = [np.expand_dims(item, axis=0) for item in inputs]
        #     self.model.train_on_batch(inputs, np.zeros((1, 1)))
        #     epoch += 1
            # if epoch % self.eval_freq == 0:
            #     print(f"# EVALUATION - EPOCH {epoch}:")
            #     hit1, metrics = self.CSLS_test()
            #     logs[epoch] = metrics
            #     if best_perf is None:
            #         best_perf = hit1
            #     elif hit1 > best_perf:
            #         best_perf = hit1
            #         no_improve_num = 0
            #     else:
            #         no_improve_num += 1
            #         if no_improve_num >= 1:
            #             break
        # with open(os.path.join(self.output_dir, "training_log.json"), "w+") as file:
        #     file.write(json.dumps(logs))


