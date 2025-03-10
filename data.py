import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm, trange
import numpy as np
from conf import parse_args

args = parse_args()
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'


class DBP15K(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.3, seed=1):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.seed = seed
        torch.manual_seed(seed)
        super(DBP15K, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self):
        return '%s_%d_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.seed)

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, self.pair[:2] + '_vectorList.json')
        x1, edge_index1, rel1, assoc1, triple1, triple1_id, t_index1, c_index1_head, h_class1, c_index1_tail, t_class1 = self.process_graph(g1_path, x1_path, emb_path)
        x2, edge_index2, rel2, assoc2, triple2, triple2_id, t_index2, c_index2_head, h_class2, c_index2_tail, t_class2 = self.process_graph(g2_path, x2_path, emb_path)

        neigh1 = generate_neighbors(triple1_id, x1.shape[0])
        neigh2 = generate_neighbors(triple2_id, x2.shape[0])

        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set, pair_set_without_trans_id = self.process_pair(pair_path, assoc1, assoc2)
        # print("pair_set", pair_set.size())
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate * pair_set.size(1))]
        test_set = pair_set[:, int(self.rate * pair_set.size(1)):]
        pair_set_without_trans_id = pair_set_without_trans_id[:, torch.randperm(pair_set.size(1))]
        train_set_without_trans_id = pair_set_without_trans_id[:, :int(self.rate * pair_set.size(1))]
        test_set_without_trans_id = pair_set_without_trans_id[:, int(self.rate * pair_set.size(1)):]
        kg1_inbound_map = self.build_inbound_map(triple1_id)
        kg2_inbound_map = self.build_inbound_map(triple2_id)
        kg1_outbound_map = self.build_outbound_map(triple1_id)
        kg2_outbound_map = self.build_outbound_map(triple2_id)

        subrel_map, sub_rel1_rel2_mtx, sub_rel2_rel1_mtx = subsumption(triple1_id, triple2_id, pair_set.t())
        func_map1, func_arr1 = self.functionality(triple1_id)
        func_map2, func_arr2 = self.functionality(triple2_id)

        if self.KG_num == 1:
            triple = triple1 + triple2
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, class_index1_head=c_index1_head, triple_index1=t_index1,
                        head_class1=h_class1, class_index1_tail=c_index1_tail, tail_class1=t_class1,neigh1=neigh1,
                        x2=x2, edge_index2=edge_index2, rel2=rel2, class_index2_head=c_index2_head, triple_index2=t_index2,
                        head_class2=h_class2, class_index2_tail=c_index2_tail, tail_class2=t_class2, neigh2=neigh2,
                        train_set=train_set.t(), test_set=test_set.t(),
                        train_set_without_trans_id=train_set_without_trans_id,
                        test_set_without_trans_id=test_set_without_trans_id,
                        tail_to_relnheads_map1=kg1_inbound_map, tail_to_relnheads_map2=kg2_inbound_map,
                        head_to_relntails_map1=kg1_outbound_map, head_to_relntails_map2=kg2_outbound_map,
                        subrel_map=subrel_map, sub_rel1_rel2_mtx=sub_rel1_rel2_mtx, sub_rel2_rel1_mtx=sub_rel2_rel1_mtx,
                        func_map1=func_map1, func_map2=func_map2, func_arr1=torch.tensor(func_arr1, device=torch.device(device)), func_arr2=torch.tensor(func_arr2, device=torch.device(device)))
            data.triple = triple
            data.triple1_id = triple1_id
            data.triple2_id = triple2_id
            data.assoc1 = assoc1
            data.assoc2 = assoc2
        else:
            x = torch.cat([x1, x2], dim=0)
            triple = triple1 + triple2
            edge_index = torch.cat([edge_index1, edge_index2 + x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2 + rel1.max() + 1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel, train_set=train_set.t(), test_set=test_set.t())
            data.triple = triple
        torch.save(self.collate([data]), self.processed_paths[0])

    def process_graph(self, triple_path, ent_path, emb_path):
        # g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        print('triple_path:', triple_path)
        g = loadfile(triple_path, 3)
        # print('g:',g)
        subj, rel, obj = torch.tensor(g).t()

        assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]

        idx = []
        with open(ent_path, 'r', encoding='utf-8') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            embedding_list = torch.tensor(json.load(f))
        x = embedding_list[idx]
        # print("idx.max().item()", idx.max().item())

        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        # print(assoc)
        print(idx.size(0))
        print(subj, obj)
        subj, obj = assoc[subj], assoc[obj]
        print(subj, obj)
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        triple_index = []
        index = 0
        hts = {}
        class_index = []
        class_dict = {}
        class_index1 = []
        class_dict1 = {}
        c = 0
        c1 = 0
        edge_index_numpy = edge_index.t().numpy()
        for i in range(len(edge_index_numpy)):
            h = edge_index_numpy[i][0]
            t = edge_index_numpy[i][1]
            r = rel[i].item()
            if (h, t) not in hts:
                hts[(h, t)] = index
                index += 1
            triple_index.append(hts[(h, t)])
            if (h, r) not in class_dict:
                class_dict[(h, r)] = c
                c += 1
            class_index.append(class_dict[(h, r)])
            if (t, r) not in class_dict1:
                class_dict1[(t, r)] = c1
                c1 += 1
            class_index1.append(class_dict1[(t, r)])

        head_class = np.zeros(c, dtype=int)
        for h, r in class_dict:
            head_class[class_dict[(h, r)]] = h

        tail_class = np.zeros(c1, dtype=int)
        for t, r in class_dict1:
            tail_class[class_dict1[(t, r)]] = t

        triple_index = torch.tensor(triple_index)
        class_index = torch.tensor(class_index)
        class_index1 = torch.tensor(class_index1)
        head_class = torch.tensor(head_class, dtype=torch.long)
        tail_class = torch.tensor(tail_class, dtype=torch.long)

        return x, edge_index, rel, assoc, g, torch.stack((subj, rel, obj)).t().numpy(), triple_index, class_index, head_class, class_index1, tail_class

    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0), torch.stack([e1, e2], dim=0)

    @staticmethod
    def build_inbound_map(triples):
        inbound_map = dict()
        for triple in triples:
            h, r, t = triple  # 确保 triples 是一个三元组列表
            if t not in inbound_map:
                inbound_map[t] = set()
            if h not in inbound_map:
                inbound_map[h] = set()
            inbound_map[t].add((r, h))  # 直接在集合上添加元素
            inbound_map[h].add((r, t))  # 直接在集合上添加元素
        inbound_map_copy = inbound_map.copy()

        # 将集合转换回列表
        for key in inbound_map:
            inbound_map[key] = list(inbound_map[key])
        print(len(inbound_map))
        return inbound_map

    @staticmethod
    def build_outbound_map(triples):
        outbound_map = dict()
        for h, r, t in triples:
            if h not in outbound_map:
                outbound_map[h] = []
            outbound_map[h].append((r, t))
        return outbound_map



    @staticmethod
    def functionality(triple_list):
        rel_to_head2tails_map = dict()
        # print(triple_list)
        for head, rel, tail in tqdm(list(triple_list)):
            # print(head, rel, tail)
            if rel not in rel_to_head2tails_map:
                rel_to_head2tails_map[rel] = {head: []}
            elif head not in rel_to_head2tails_map[rel]:
                rel_to_head2tails_map[rel][head] = []
            rel_to_head2tails_map[rel][head].append(tail)

        func_map = {}
        for rel in tqdm(rel_to_head2tails_map.keys(), desc="computing functionality"):
            head2tails_map = rel_to_head2tails_map[rel]
            num_head = len(head2tails_map)
            num_head_tail_pair = 0
            for head in head2tails_map.keys():
                num_head_tail_pair += len(head2tails_map[head])
            func = float(num_head) / float(num_head_tail_pair)
            func_map[rel] = func
        rel_num = len(func_map)
        print("rel_num:------------",rel_num)
        func_list = [func_map[rel] for rel in range(rel_num)]
        return func_map, func_list

def subsumption(triple1_list, triple2_list, alignment):
    ent2_to_ent1_map = dict()
    for ent1, ent2 in alignment:
        ent2_to_ent1_map[ent2] = ent1

    # assign id to each entity
    ent1_list = []
    rel1_list = []
    for head, rel, tail in triple1_list:
        ent1_list.extend([head, tail])
        rel1_list.append(rel)
    ent1_list = list(set(ent1_list))
    rel1_list = sorted(list(set(rel1_list)))
    ent_to_no_map = dict()
    for idx, ent in enumerate(ent1_list):
        ent_to_no_map[ent] = idx
    ent2_list = []
    rel2_list = []
    for head, rel, tail in triple2_list:
        ent2_list.extend([head, tail])
        rel2_list.append(rel)
    ent2_list = list(set(ent2_list))
    rel2_list = sorted(list(set(rel2_list)))
    ent_no_cursor = len(ent1_list)
    for ent2 in ent2_list:
        if ent2 in ent2_to_ent1_map:
            cor_ent1 = ent2_to_ent1_map[ent2]
            ent_to_no_map[ent2] = ent_to_no_map[cor_ent1]
        else:
            ent_to_no_map[ent2] = ent_no_cursor
            ent_no_cursor += 1

    # count
    rel_to_ent_map1 = dict()
    for head, rel, tail in tqdm(triple1_list):
        if rel not in rel_to_ent_map1:
            rel_to_ent_map1[rel] = []
        rel_to_ent_map1[rel].append((ent_to_no_map[head], ent_to_no_map[tail]))

    rel_to_ent_map2 = dict()
    for head, rel, tail in tqdm(triple2_list):
        if rel not in rel_to_ent_map2:
            rel_to_ent_map2[rel] = []
        rel_to_ent_map2[rel].append((ent_to_no_map[head], ent_to_no_map[tail]))

    subrel_map = dict()
    sub_rel1_rel2_mtx = np.zeros(shape=(len(rel1_list), len(rel2_list)), dtype=np.float16)
    sub_rel2_rel1_mtx = np.zeros(shape=(len(rel2_list), len(rel1_list)), dtype=np.float16)
    for rel1 in tqdm(rel1_list, desc="count subrelation"):
        ent_pair_set1 = set(rel_to_ent_map1[rel1])
        for rel2 in rel2_list:
            ent_pair_set2 = set(rel_to_ent_map2[rel2])
            interset = ent_pair_set1.intersection(ent_pair_set2)
            rel1_rel2 = len(interset) / len(ent_pair_set1)
            rel2_rel1 = len(interset) / len(ent_pair_set2)
            subrel_map[(f"kg1:{rel1}", f"kg2:{rel2}")] = rel1_rel2
            subrel_map[(f"kg2:{rel2}", f"kg1:{rel1}")] = rel2_rel1
            sub_rel1_rel2_mtx[rel1][rel2] = rel1_rel2
            sub_rel2_rel1_mtx[rel2][rel1] = rel2_rel1
    del ent2_to_ent1_map
    del ent_to_no_map
    del rel_to_ent_map1
    del rel_to_ent_map2
    return subrel_map, torch.tensor(sub_rel1_rel2_mtx, device=torch.device(device)), torch.tensor(sub_rel2_rel1_mtx, device=torch.device(device))
def load_relation(triples):
    # 初始化一个全零的二维列表，行数为e_num，列数为e_num
    # relation_List = [[0 for _ in range(e_num)] for _ in range(e_num)]
    neigh = dict()
    r_dict = dict()
    # print("triples:",triples[:200])
    triples = list(triples)
    for h, r, t in triples:

        if h not in neigh:
            # print(h)
            neigh[h] = set()
        neigh[h].add(t)
        # else:
        #     # if t not in neigh[h]:
        #     neigh[h].add(t)
        if t not in neigh:
            neigh[t] = set()
        neigh[t].add(h)
        # print(neigh)
        # else:
        #     # if h not in neigh[t]:
        #     neigh[t].add(h)

        # 检查relation_List[h][t]是否为0，如果是，说明这个位置还没有关系，需要初始化为列表
        if (h, t) not in r_dict:
            r_dict[(h, t)] = [r]
        else:
            # 如果已经有了关系，就追加到列表中
            r_dict[(h, t)].append(r)
        if h != t:
            if (t, h) not in r_dict:
                r_dict[(t, h)] = [r]
            else:
                # 如果已经有了关系，就追加到列表中
                r_dict[(t, h)].append(r)

    # # 将relation_List转换为PyTorch张量
    # relation_List_tensor = torch.tensor(relation_List)
    #
    # # 将张量移动到GPU上
    # relation_List_tensor = relation_List_tensor.cuda(0)

    print("加载关系完毕!")
    # print(neigh[23603])

    return r_dict, neigh

def funcR(triple):
    prr = {}
    heads = {}
    tails = {}

    for (h, r, t) in triple:
        if r not in prr:
            prr[r] = 1
            heads[r] = [h]
            tails[r] = [t]
        else:
            prr[r] +=1
            if h not in heads[r]:
                heads[r].append(h)
            if t not in tails[r]:
                tails[r].append(t)
    fr = {}
    frn = {}
    for r in prr:
        fr[r] = len(heads[r])/prr[r]
        frn[r] = len(tails[r])/prr[r]

    return fr, frn

def loadfile(fn, num=1):
    print('正在读取文件：' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.strip().split('\t')
            x = []
            neigh = {}
            for i in range(num):
                x.append(int(th[i]))

            # 存储三元组
            ret.append(tuple(x))
    return ret

def generate_class(edge_index, rel_class_id):
    triple_index = []
    index = 0
    hts = {}
    class_index_head = []
    class_dict = {}
    class_index_tail = []
    class_dict1 = {}
    c = 0
    c1 = 0
    edge_index_numpy = edge_index.t().numpy()
    for i in range(len(edge_index_numpy)):
        h = edge_index_numpy[i][0]
        t = edge_index_numpy[i][1]
        r = rel_class_id[i].item()
        if (h, t) not in hts:
            hts[(h, t)] = index
            index += 1
        triple_index.append(hts[(h, t)])
        if (h, r) not in class_dict:
            class_dict[(h, r)] = c
            c += 1
        class_index_head.append(class_dict[(h, r)])
        if (t, r) not in class_dict1:
            class_dict1[(t, r)] = c1
            c1 += 1
        class_index_tail.append(class_dict1[(t, r)])

    head_class = np.zeros(c, dtype=int)
    for h, r in class_dict:
        head_class[class_dict[(h, r)]] = h

    tail_class = np.zeros(c1, dtype=int)
    for t, r in class_dict1:
        tail_class[class_dict1[(t, r)]] = t

    triple_index = torch.tensor(triple_index)
    class_index_head = torch.tensor(class_index_head)
    class_index_tail = torch.tensor(class_index_tail)
    head_class = torch.tensor(head_class, dtype=torch.long)
    tail_class = torch.tensor(tail_class, dtype=torch.long)
    return triple_index, class_index_head, head_class, class_index_tail, tail_class

def generate_neighbors(triple, ent_num):
    # print("ent_num", ent_num)
    # neighbors_h = {}
    # neighbors_t = {}
    neighbors = {}
    for h, r, t in triple:
        if h == -1 or t ==-1:
            print(h, r, t)
        if h not in neighbors:
            neighbors[h] = set()
        neighbors[h].add(t)
        if t not in neighbors:
            neighbors[t] = set()
        neighbors[t].add(h)
    # neighbors_h_copy = neighbors_h.copy()
    # neighbors_t_copy = neighbors_t.copy()
    # for h, r, t in tqdm(triple, desc="Generating head and tail neighbors"):
    #     if t in neighbors_h_copy:
    #         neighbors_h[h].update(neighbors_h_copy[t])
    #     if h in neighbors_t_copy:
    #         neighbors_t[t].update(neighbors_t_copy[h])
    # neighbors = dict()
    # print(neighbors.keys())
    for i in trange(len(neighbors.keys()), desc="Generating one hop neighbors"):
        # print(neighbors_h[i], neighbors_t[i])
        # neighbors[i] = neighbors_h.get(i, set()) | neighbors_t.get(i, set())
        # print(neighbors[i])
        neighbors[i] = list(neighbors[i])
    return neighbors

