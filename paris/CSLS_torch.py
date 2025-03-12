import multiprocessing

import gc
import os

import numpy as np
import time
import torch
from tqdm import trange
import torch.nn.functional as F



class Evaluator():
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.batch_size = 512

    # csls distance
    def csls_sim(self, embed1: np.ndarray, embed2: np.ndarray, k=10):
        t1 = time.time()
        # Calculate the Cosine distance matrix (n*m)
        sim_mat = self.cosine_sim(embed1, embed2)

        if k <= 0:
            print("k = 0")
            return sim_mat
        # Calculate the average of the top k similarities that are closest to each other
        csls1 = self.CSLS_thr(sim_mat, k)
        csls2 = self.CSLS_thr(sim_mat.T, k)

        # target：2*sim_mat - csls1 - csls2
        csls_sim_mat = self.compute_csls(sim_mat, csls1, csls2)

        t2 = time.time()
        print(f"sim handler spends time: {t2 - t1}s")
        return csls_sim_mat

    # target：2*sim_mat - csls1 - csls2
    def compute_csls(self, sim_mtx: np.ndarray, row_thr:np.ndarray, col_thr:np.ndarray):
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            col_thr = torch.tensor(col_thr, device=self.device).unsqueeze(dim=0)
            for cursor in trange(0, total_size, self.batch_size, desc="csls metrix"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_row_thr = row_thr[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                sub_row_thr = torch.tensor(sub_row_thr, device=self.device)
                sub_sim_mtx = 2*sub_sim_mtx - sub_row_thr.unsqueeze(dim=1) - col_thr
                sim_mtx[cursor: cursor+self.batch_size] = sub_sim_mtx.cpu().numpy()

        return sim_mtx

    # Cosine distance between KG1 and KG2
    def cosine_sim(self, embed1, embed2):
        with torch.no_grad():
            total_size = embed1.shape[0]
            embed2 = embed2.t()
            sim_mtx= np.empty(shape=(embed1.shape[0], embed2.shape[1]), dtype=np.float32)
            for cursor in trange(0, total_size, self.batch_size, desc="cosine matrix"):
                sub_embed1 = embed1[cursor: cursor + self.batch_size]
                sub_sim_mtx = torch.matmul(sub_embed1, embed2)
                sim_mtx[cursor: cursor + self.batch_size] = sub_sim_mtx.cpu().numpy()
        return sim_mtx

    # CSLS -- the average of the top k similarities that are closest to each other
    def CSLS_thr(self, sim_mtx: np.ndarray, k=10):
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            sim_value_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="csls thr"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                nearest_k, _ = torch.topk(sub_sim_mtx, dim=1, k=k, largest=True, sorted=False)
                sim_values = nearest_k.mean(dim=1, keepdim=False)
                sim_value_list.append(sim_values.cpu().numpy())
            sim_values = np.concatenate(sim_value_list, axis=0)
        return sim_values


