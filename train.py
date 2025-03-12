
import itertools

import apex
import torch

import torch.nn.functional as F
from conf import parse_args
from label_generate import enhance_latent_labels_with_DAA, enhance_latent_labels_with_OSS
from model import DSGEA
from data import DBP15K
from loss import L1_Loss
from paris.simi_to_prob import SimiToProbModule
from utils import add_inverse_rels, get_train_batch, get_hits


def init_data(args, device):
    # initialize data
    data = DBP15K(args.data, args.lang, rate=args.rate)[0]

    # normalization
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()

    # reverse the direction of the relationship
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)

    data.original_set = data.train_set.detach().clone()
    data.temp_set = data.train_set.detach().clone()
    data.new_set = data.train_set.detach().clone()
    return data

# get entity embeddings
def get_emb(model, data, classify):
    model.eval()
    with torch.no_grad():
        # integrate the classifier
        if classify:
            # KG1 embeddings
            x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, 0)
            # KG2 embeddings
            x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel1.max().item() + 1)
        # exclude the classifier
        else:
            # KG1 embeddings
            x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.triple_index1, data.class_index1_head,
                       data.head_class1, data.class_index1_tail, data.tail_class1, 0)
            # KG2 embeddings
            x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.triple_index2,data.class_index2_head,
                       data.head_class2, data.class_index2_tail, data.tail_class2, data.rel1.max().item()+1)
    return x1, x2

# train the model(embedding)
def train(model, criterion, optimizer, data, classify):
    model.train()

    # integrate the classifier
    if classify:
        # KG1 embeddings
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, 0)
        # KG2 embeddings
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel1.max().item() + 1)
    # exclude the classifier
    else:
        # KG1 embeddings
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.triple_index1,
                   data.class_index1_head, data.head_class1, data.class_index1_tail, data.tail_class1, 0)
        # KG2 embeddings
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.triple_index2,
                   data.class_index2_head,
                   data.head_class2, data.class_index2_tail, data.tail_class2, data.rel1.max().item() + 1)
    # calculate the loss
    loss = criterion(x1, x2, data.train_set, data.train_batch)
    # backpropagation/train
    optimizer.zero_grad()
    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    return loss

# calculate the hit rate on the test set
def test(model, data):
    x1, x2 = get_emb(model, data, args.classify)
    # calculate the hit rate
    test_hits_left_1, test_hits_left_10, mrr = get_hits(x1, x2, data.test_set)
    return test_hits_left_1, test_hits_left_10, mrr

# train the DSGEA
def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # acquire and initialize data
    data = init_data(args, device).to(device)

    # initialize model
    model = DSGEA(data.x1.size(1), args.r_hidden, args.c_hidden, data.rel1.max().item()+data.rel2.max().item()+2).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    model, optimizer = apex.amp.initialize(model, optimizer)
    criterion = L1_Loss(args.gamma)

    # store the candidate pairs
    data.new_set = torch.tensor([]).to(device)

    # initialize the number of iterations
    if args.lang == 'ga_en':
        intersect = 1
        num = 6
    else:
        intersect = int(args.intersect)
        num = int(args.repeat1)

    # iteratively train model
    for ite in range(args.em_iteration_num):
        print("------------ite", ite, "------------")

        times = 0
        max_hits = 0

        # add the candidates to the training set every n iterations
        if ite > 0 and (ite + 1) % intersect == 0:
            assert data.new_set.shape[0] > 0
            data.temp_set = data.new_set
            data.train_set = torch.cat((data.train_set, data.new_set))
            assert data.train_set.shape[0] > 0
            data.new_set = torch.tensor([]).to(device)
        if ite % num == 0:
            data.train_set = data.original_set
            data.temp_set = data.original_set
            data.new_set = torch.tensor([]).to(device)
        if ite == num:
            num = int(args.repeat2)

        # train the embeddings using training data.
        for epoch in range(args.epoch):
            if epoch % args.neg_epoch == 0:
                x1, x2 = get_emb(model, data, args.classify)
                data.train_batch = get_train_batch(x1, x2, data.train_set, args.k)
            # backpropagate based on the loss function
            loss = train(model, criterion, optimizer, data, args.classify)

            print('Epoch:', epoch + 1, '/', args.epoch, '\tLoss: %.3f' % loss, '\r', end='')
            # stop current round of training when the number of times the hit rate decreases exceeds 3.
            if (epoch + 1) % args.test_epoch == 0:
                test_hits1_left, test_hits10_left, mrr = test(model, data)
                if test_hits1_left > max_hits:
                    max_hits = test_hits1_left
                else:
                    times += 1
            if ite > 0 and times >= 2:
                break

        # calculate the similarity
        x1, x2 = get_emb(model, data, args.classify)
        simi_mtx = model.predict_simi(x1, x2, device)
        simi2prob_model = SimiToProbModule(device, args.output)

        # similarity training
        simi2prob_model.train_model(simi_mtx, data.train_set, data.test_set)
        simi2prob_model_inv = SimiToProbModule(device, args.output, inv=True)
        simi2prob_model_inv.train_model(simi_mtx, data.train_set, data.test_set)
        neural_sim_mtx = None


        # compute the similarity matrix
        if args.joint_distri_model:
            neural_prob_mtx, neural_prob_mtx_inv = model.predict(x1, x2, device)

        # search for candidate pairs
        if args.DAA:
            # adopt the DAA method
            enhance_latent_labels_with_DAA(x1.cpu(), x2.cpu(), neural_prob_mtx, neural_prob_mtx_inv, neural_sim_mtx, data)
        else:
            # adopt the optimal selection strategy
            enhance_latent_labels_with_OSS(x1.cpu(), x2.cpu(), neural_prob_mtx, neural_prob_mtx_inv, neural_sim_mtx, data)

        # calculate the hit rate and MRR
        if ite == args.em_iteration_num - 1 or ite == 0:
            test_hits1_left, test_hits10_left, mrr = test(model, data)
            print('Hits@1: %.2f%%    ' % test_hits1_left, end='')
            print('Hits@10: %.2f%%    ' % test_hits10_left, end='')
            print('MRR: %.3f' % mrr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
