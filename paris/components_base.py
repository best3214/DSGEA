
import numpy as np

from paris.simi_to_prob import SimiToProbModule


def convert_simi_to_probs7(simi_mtx: np.ndarray, device, output_dir, inv=False):  # SimiToProbModel
    simi2prob_module = SimiToProbModule(device, output_dir)
    if inv:
        probs = simi2prob_module.predict(simi_mtx.transpose())
    else:
        probs = simi2prob_module.predict(simi_mtx)
    # if not probs.is_cuda:
    #     probs = probs.to(device)
    return probs




def compute_metrics(ranking_list, rels):
    gold_rank_list = []
    for idx, rel in enumerate(rels):
        ranking = ranking_list[idx]
        gold_rank = np.where(ranking == rel)[0][0]
        gold_rank_list.append(gold_rank+1)
    gold_rank_arr = np.array(gold_rank_list)
    mean_rank = np.mean(gold_rank_arr)
    mrr = np.mean(1.0/gold_rank_arr)
    recall_1 = np.mean((gold_rank_arr <= 1).astype(np.float32))
    recall_5 = np.mean((gold_rank_arr <= 5).astype(np.float32))
    recall_10 = np.mean((gold_rank_arr <= 10).astype(np.float32))
    recall_50 = np.mean((gold_rank_arr <= 50).astype(np.float32))
    return mean_rank, mrr, recall_1, recall_5, recall_10, recall_50


def evaluate_models(score_mtx: np.ndarray, eval_alignment):
    eval_alignment_arr = np.array(eval_alignment)
    rel_score_mtx = score_mtx[eval_alignment_arr[:, 0]][:, eval_alignment_arr[:, 1]]
    pred_ranking = np.argsort(-rel_score_mtx, axis=1)
    pred_ranking = eval_alignment_arr[:, 1][pred_ranking]
    rels = eval_alignment_arr[:, 1]
    mr, mrr, recall_1, recall_5, recall_10, recall_50 = compute_metrics(pred_ranking, rels)
    print(f"mr:{mr}, mrr:{mrr}, recall@1:{recall_1}, recall@5:{recall_5}, recall@10:{recall_10}, recall@50:{recall_50}")
    metrics = {
        "mr": float(mr),
        "mrr": float(mrr),
        "recall@1": float(recall_1),
        "recall@5": float(recall_5),
        "recall@10": float(recall_10),
        "recall@50": float(recall_50)
    }
    return metrics




