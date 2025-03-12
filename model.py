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


# embedding implementation
class DSGEA(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100, c_hidden=150, r_num=None, c_num=150):
        super(DSGEA, self).__init__()
        # layers of GCN
        self.gcn1 = GCN(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.gcn3 = GCN(e_hidden)
        self.gcn4 = GCN(e_hidden)
        self.gcn5 = GCN(e_hidden)

        # layers of highway
        self.highway1 = Highway(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.highway3 = Highway(e_hidden)
        self.highway4 = Highway(e_hidden)

        # layer of relation embeddings
        self.gat_e_to_r = GAT_E_to_R(e_hidden, c_hidden, r_hidden)
        # layer of entity embedding
        self.gat_e_T = GAT_E_T(e_hidden, c_hidden, r_hidden)

        self.gat = GAT(e_hidden * 2)
        self.gat1 = GAT(e_hidden)

        # classifier
        self.classifier = Classifier(r_hidden, int(r_hidden / 2), c_num)
        if r_num is not None:
            self.r_emb = nn.Embedding(r_num, r_hidden)

    # implement category-based splitting and aggregating entity embedding
    def forward(self, x_e, edge_index, rel, edge_index_all, *args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # integrate the classifier
        if len(args) == 1:
            start_num = args[0]

            # random initialization of relation embeddings
            r_emb = F.normalize(self.r_emb(torch.arange(start_num, rel.max()+1+start_num).to(device)), p=2, dim=1)

            # relation classification
            r_class = self.classifier(r_emb)
            self.r_class_id = r_class[rel]

            triple_num, class_index_head, head_class, class_index_tail, tail_class = generate_class(edge_index.cpu(), self.r_class_id.cpu())
            triple_num, class_index_head, head_class, class_index_tail, tail_class = triple_num.to(device), class_index_head.to(device), head_class.to(device), class_index_tail.to(device), tail_class.to(device)

            # highway + GCN
            x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
            x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))

            # generate relation embeddings
            x_e_h = self.gat_e_to_r(x_e, edge_index, rel, triple_num, r_emb, class_index_head, head_class)

            # category-based splitting and aggregating entity embedding
            x_e_h_t = torch.cat([x_e_h, self.gat_e_T(x_e, edge_index, rel, triple_num, r_emb, class_index_tail, tail_class)], dim=1)

            # generate ultimate entity embeddings by concatenating
            x_e1 = torch.cat([x_e, x_e_h_t], dim=1)
            x_e = torch.cat([x_e, self.gat(x_e1, edge_index_all)], dim=1)
        # exclude the classifier
        else:
            triple_num, class_index_head, head_class, class_index_tail, tail_class, start_num = args
            # random initialization of relation embeddings
            r_emb = F.normalize(self.r_emb(torch.arange(start_num, rel.max() + 1 + start_num).to(device)), p=2, dim=1)

            # highway + GCN
            x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
            x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))

            # generate relation embeddings
            x_e_h = self.gat_e_to_r(x_e, edge_index, rel, triple_num, r_emb, class_index_head, head_class)
            # category-based splitting and aggregating entity embedding
            x_e_h_t = torch.cat(
                [x_e_h, self.gat_e_T(x_e, edge_index, rel, triple_num, r_emb, class_index_tail, tail_class)], dim=1)

            # generate ultimate entity embeddings by concatenating
            x_e1 = torch.cat([x_e, x_e_h_t], dim=1)
            x_e = torch.cat([x_e, self.gat(x_e1, edge_index_all)], dim=1)
        return x_e

    # classifier
    def classify(self, x_r):
        r_class_id = self.classifier(x_r)
        return r_class_id


    # compute embedding distance by CSLS
    def predict_simi(self, ent1_embs, ent2_embs, device):

        evaluator = Evaluator(device=device)
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs)

        return simi_mtx

    # translate embedding distance into similarity
    def predict(self, x1, x2, device, no_sim2prob=False):
        # csls distance
        simi_mtx = self.predict_simi(x1, x2, device)

        # calculate similarity
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

# Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

# the gating mechanism
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x

# Relation Embedding
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
        x_r_h = F.relu(self.w_h(x_e))
        x_r_t = F.relu(self.w_t(x_e))

        # （1）Split the triplet
        # Calculate the split factor -- alpha
        e1 = (self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_t1(x_r_t).squeeze()[edge_index_t]) / 2
        e_r = self.a_r1(r_emb).squeeze()[rel]
        e1 = e1 + e_r
        alpha = softmax(F.leaky_relu(e1).float(), triple_num)

        # （2）Aggregate by category
        # 2.1 Calculate the Aggregation Degree of Tail Entities -- e_t
        x_t = x_r_t[edge_index_t]
        x_t = x_t * alpha.view(-1, 1)
        x_t = F.normalize(x_t, dim=1, p=2).requires_grad_()
        e_t = self.a_t2(x_t).squeeze()

        # 2.2 Calculate the Aggregation Degree of Head Entities -- e_h
        e_h = self.a_h2(x_r_h).squeeze()[edge_index_h]

        # 2.3 Calculate the Aggregation Degree of Relations -- e_r
        e_r = self.a_r2(r_emb).squeeze()[rel]
        e_r_e_h = self.a_h3(x_r_h).squeeze()[edge_index_h]
        e_r_e_t = self.a_t3(x_t).squeeze()
        e_r = (e_r + (e_r_e_h + e_r_e_t) / 2) / 2

        # 2.4 Calculate the Classification Coefficient -- beta
        e = e_t + e_r + e_h
        beta = softmax(F.leaky_relu(e).float(), class_index)

        # 2.5 Category Embeddings -- x_class
        x_class = spmm(torch.cat([class_index.view(1, -1), edge_index_t.view(1, -1)], dim=0), beta, class_index.max() + 1, x_t.size(0),
                     x_t)

        # （3）Aggregate Category Embeddings
        e_c = self.a_c(x_class).squeeze()
        e_h_c = self.a_h4(x_r_h).squeeze()[head_class]
        e_c = e_c + e_h_c

        gama = softmax(F.leaky_relu(e_c).float(), head_class)
        class_num = torch.arange(class_index.max() + 1).to(x_t.device)
        x_e_h = spmm(torch.cat([head_class.view(1, -1), class_num.view(1, -1)], dim=0), gama, x_e.size(0), class_index.max() + 1,
                     x_class)
        x_e_h = self.highway(x_r_h, x_e_h)

        return x_e_h

# Entity Embedding
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

        # （1）Split the triplet
        # Calculate the split factor -- alpha
        e1 = (self.a_h1(x_r_h).squeeze()[edge_index_h] + self.a_t1(x_r_t).squeeze()[edge_index_t]) / 2
        e_r = self.a_r1(r_emb).squeeze()[rel]
        e1 = e1 + e_r
        alpha = softmax(F.leaky_relu(e1).float(), triple_num)

        # （2）Aggregate by category
        # 2.1 Calculate the Aggregation Degree of Head Entities -- e_h
        x_h = x_r_h[edge_index_h]
        x_h = x_h * alpha.view(-1, 1)
        x_h = F.normalize(x_h, dim=1, p=2).requires_grad_()
        e_h = self.a_h2(x_h).squeeze()

        # 2.2计算尾实体聚合度e_t
        # 2.2 Calculate the Aggregation Degree of Tail Entities -- e_t
        e_t = self.a_t2(x_r_t).squeeze()[edge_index_t]

        # 2.3 Calculate the Aggregation Degree of Relations -- e_r
        e_r = self.a_r2(r_emb).squeeze()[rel]
        e_r_e_t = self.a_t3(x_r_t).squeeze()[edge_index_t]
        e_r_e_h = self.a_h3(x_h).squeeze()
        e_r = (e_r + (e_r_e_h + e_r_e_t) / 2) / 2

        # 2.4 Calculate the Classification Coefficient -- beta
        e = e_t + e_r + e_h
        beta = softmax(F.leaky_relu(e).float(), class_index)

        # 2.5 Category Embeddings -- x_class
        x_class = spmm(torch.cat([class_index.view(1, -1), edge_index_h.view(1, -1)], dim=0), beta, class_index.max() + 1, x_h.size(0),
                     x_h)

        # （3）Aggregate Category Embeddings
        e_c = self.a_c(x_class).squeeze()
        e_t_c = self.a_h4(x_r_t).squeeze()[tail_class]
        e_c = e_c + e_t_c

        gama = softmax(F.leaky_relu(e_c).float(), tail_class)
        class_num = torch.arange(class_index.max() + 1).to(x_e.device)

        x_e_t = spmm(torch.cat([tail_class.view(1, -1), class_num.view(1, -1)], dim=0), gama, x_e.size(0), class_index.max() + 1,
                     x_class)
        x_e_t = self.highway(x_r_t, x_e_t)

        return x_e_t
    
# GAT Network
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
    




