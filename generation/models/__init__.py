import torch
import torchvision.models as models

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from . import vae, deepinfomax
import time

__version__ = "0.7.0"
from .model import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)


try:
    from apex import amp
except:
    pass

def get_model(exp_dict, **kwargs):
    if exp_dict["model"] == 'vae':
        return vae.VAE(exp_dict, **kwargs) 
    elif exp_dict["model"] == "deepinfomax":
        return deepinfomax.DeepInfoMax(exp_dict, **kwargs)
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])
        
