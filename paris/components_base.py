
import numpy as np

from paris.simi_to_prob import SimiToProbModule

# Similarity Transformation
def convert_simi_to_probs7(simi_mtx: np.ndarray, device, output_dir, inv=False):  # SimiToProbModel
    simi2prob_module = SimiToProbModule(device, output_dir)
    if inv:
        probs = simi2prob_module.predict(simi_mtx.transpose())
    else:
        probs = simi2prob_module.predict(simi_mtx)
    return probs










