
import os
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# the detail of calculating Similarity

# Normalize Similarity
class Simi2ProbModel(nn.Module):
    def __init__(self, fea_num):
        super(Simi2ProbModel, self).__init__()
        self.linear = nn.Linear(in_features=fea_num, out_features=1)
        self.crx_ent_tau = nn.Parameter(data=torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

    def forward(self, simi_mtx: torch.Tensor, max_simi_arr: torch.Tensor):  # N, cate_num, fea_dim
        offset = max_simi_arr.unsqueeze(dim=1) - simi_mtx
        features = torch.stack([simi_mtx, offset], dim=-1)
        logits = self.linear(features) / self.crx_ent_tau
        logits = torch.squeeze(logits)
        return logits


# related Similarity data
class Simi2ProbDataset(Dataset):
    def __init__(self, simi_mtx, max_simi_arr, labels=None):
        self.simi_mtx = simi_mtx
        self.max_simi_arr = max_simi_arr
        self.labels = labels

    def __getitem__(self, idx):
        if self.labels is None:
            return torch.tensor(self.simi_mtx[idx], dtype=torch.float32), \
                torch.tensor(self.max_simi_arr[idx], dtype=torch.float32)
        else:
            return torch.tensor(self.simi_mtx[idx], dtype=torch.float32), \
                torch.tensor(self.max_simi_arr[idx], dtype=torch.float32), \
                torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.simi_mtx)


class SimiToProbModule:
    def __init__(self, device, output_dir=None, inv=False):
        # self.conf = conf

        self.device = torch.device(device)
        self.model = Simi2ProbModel(fea_num=2)
        self.model.to(self.device)
        self.inv = inv
        self.output_dir = output_dir


    # Similarity Training
    def train_model(self, simi_mtx, train_alignment, test_alignment):
        if self.inv:

            print("generating inv_features")
            test_alignment = test_alignment.cpu().numpy()
            train_alignment = train_alignment.cpu().numpy()
            np.random.shuffle(test_alignment)
            test_alignment = test_alignment[:100]
            train_features, train_labels = self.generate_features(simi_mtx.transpose(1, 0),
                                                                  train_alignment[:, [1, 0]]), train_alignment[:, 0]
            test_features, test_labels = self.generate_features(simi_mtx.transpose(1, 0),
                                                                test_alignment[:, [1, 0]]), test_alignment[:, 0]

            dataset = Simi2ProbDataset(train_features[0], train_features[1], train_labels)
            dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=64)
            test_dataset = Simi2ProbDataset(test_features[0], test_features[1], test_labels)
            test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64)

            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
            lowest_loss = None
            no_dec_step = 0

            for epoch in range(100):
                loss_item_list = []
                for batch in dataloader:
                    simi_mtx, max_simi_arr, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                    logits = self.model(simi_mtx, max_simi_arr)

                    loss = F.cross_entropy(input=logits, target=labels, reduction="mean")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.detach().cpu().item()
                    loss_item_list.append(loss_item)
                ave_loss_item = np.mean(loss_item_list)

                eval_loss_item_list = []
                with torch.no_grad():
                    for batch in test_dataloader:
                        simi_mtx, max_simi_arr, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                        logits = self.model(simi_mtx, max_simi_arr)
                        loss = F.cross_entropy(input=logits, target=labels, reduction="mean")
                        loss_item = loss.detach().cpu().item()
                        eval_loss_item_list.append(loss_item)
                ave_eval_loss_item = np.mean(eval_loss_item_list)


                if lowest_loss is None or ave_eval_loss_item < lowest_loss:
                    lowest_loss = ave_eval_loss_item
                    no_dec_step = 0
                    self.save()
                else:
                    no_dec_step += 1
                    if no_dec_step >= 3:
                        break
        else:
            print("generating features")

            test_alignment = test_alignment.cpu().numpy()
            train_alignment = train_alignment.cpu().numpy()
            np.random.shuffle(test_alignment)
            test_alignment = test_alignment[:100]

            train_features, train_labels = self.generate_features(simi_mtx, train_alignment), train_alignment[:, 1]
            test_features, test_labels = self.generate_features(simi_mtx, test_alignment), test_alignment[:, 1]

            dataset = Simi2ProbDataset(train_features[0], train_features[1], train_labels)
            dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=64)
            test_dataset = Simi2ProbDataset(test_features[0], test_features[1], test_labels)
            test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64)

            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
            lowest_loss = None
            no_dec_step = 0
            for epoch in range(100):
                loss_item_list = []
                for batch in dataloader:
                    simi_mtx, max_simi_arr, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(
                        self.device)
                    logits = self.model(simi_mtx, max_simi_arr)
                    loss = F.cross_entropy(input=logits, target=labels, reduction="mean")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.detach().cpu().item()
                    loss_item_list.append(loss_item)
                ave_loss_item = np.mean(loss_item_list)

                eval_loss_item_list = []
                with torch.no_grad():
                    for batch in test_dataloader:
                        simi_mtx, max_simi_arr, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                        logits = self.model(simi_mtx, max_simi_arr)
                        loss = F.cross_entropy(input=logits, target=labels, reduction="mean")

                        loss_item = loss.detach().cpu().item()
                        eval_loss_item_list.append(loss_item)
                ave_eval_loss_item = np.mean(eval_loss_item_list)


                if lowest_loss is None or ave_eval_loss_item < lowest_loss:
                    lowest_loss = ave_eval_loss_item
                    no_dec_step = 0
                    self.save()
                else:
                    no_dec_step += 1
                    if no_dec_step >= 3:
                        break

    # predict similarity
    def predict(self, simi_mtx: np.ndarray):
        self.load()
        simi_mtx, max_simi_arr = self.generate_features(simi_mtx)
        dataset = Simi2ProbDataset(simi_mtx, max_simi_arr)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=64)
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="simi2prob"):
                batch_simi_mtx, batch_max_simi = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(batch_simi_mtx, batch_max_simi)
                probs = F.softmax(logits, dim=1)
                simi_mtx[cursor:cursor + len(probs)] = probs.cpu().numpy()
                cursor += len(probs)
        return simi_mtx

    # Feature Extraction
    def generate_features(self, simi_mtx: np.ndarray, alignment: np.ndarray = None):


        if alignment is None:
            sub_simi_mtx = simi_mtx
        else:
            if isinstance(alignment, torch.Tensor):
                alignment = alignment.cpu().numpy()
            sub_simi_mtx = simi_mtx[alignment[:, 0]]

        batchsize = 512
        total_size = sub_simi_mtx.shape[0]

        max_simi_list = []
        with torch.no_grad():
            for cursor in trange(0, total_size, batchsize, desc="generate features"):
                batch_simi_mtx = torch.tensor(sub_simi_mtx[cursor:cursor + batchsize], device=self.device)
                batch_max_simi, _ = torch.max(batch_simi_mtx, dim=1, keepdim=False)
                max_simi_list.append(batch_max_simi.cpu().numpy())

        max_simi_arr = np.concatenate(max_simi_list, axis=0)

        return sub_simi_mtx, max_simi_arr
