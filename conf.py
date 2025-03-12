

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--continue_training", default="sup")



    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--joint_distr_thr", type=float, default=0)

    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--c_hidden", type=int, default=150)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--intersect", type=int, default=4)
    parser.add_argument("--repeat1", type=int, default=20)
    parser.add_argument("--repeat2", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=3)

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--sim_epoch", type=int, default=10)
    parser.add_argument("--em_iteration_num", type=int, default=1)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=5)

    parser.add_argument("--joint_distri_model", type=bool, default=True)
    parser.add_argument("--classify", type=bool, default=True)
    parser.add_argument("--DAA", type=bool, default=True)
    parser.add_argument("--no_joint_distr", type=bool, default=False)
    parser.add_argument("--joint_distri_model_inv", type=bool, default=True)
    parser.add_argument("--improved_candi_probs", type=bool, default=False)
    args = parser.parse_args()
    return args
