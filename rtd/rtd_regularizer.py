import numpy as np
from scipy.special import softmax
import torch

from rtd.rtd import MinMaxRTDLoss, z_dist

normal_s = lambda x: 0.5 * (torch.erf(x/np.sqrt(2)) + 1)
normal_sinv = lambda x: np.sqrt(2) * torch.erfinv(2 * x - 1)

class RTDRegularizer:
    def __init__(self, lp, q_normalize):
        self.rtd = MinMaxRTDLoss(dim=1, lp=lp,  **{"engine":"ripser", "is_sym":True, "card":50})
        self.q_normalize = q_normalize

    def compute_reg(self, mask_source, mask_augmentation):
        cloud1, q11, q12 = z_dist(mask_source.flatten(1, -1), q_normalize=self.q_normalize)
        cloud2, q21, q22 = z_dist(mask_augmentation.flatten(1, -1), q_normalize=self.q_normalize)
        *_, rtd_loss = self.rtd(cloud1, cloud2)
        return rtd_loss
