import torch
import os
from torch import cuda
from src.utils import Color_map
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class config(object):

  model_path = "./model/"
  path = "./Dataset/"
  load_model = "./model/state_dict.pt" #best performing model using lr= 0.0008, epoch = 50
  batch = 1
  lr = 0.0008
  epochs = 50
  input_size = (180,320)
  if torch.backends.mps.is_available(): device = torch.device("mps")
  else: device = torch.device('cpu')
  code2id, id2code, name2id, id2name = Color_map(path+'class_dict.csv')
