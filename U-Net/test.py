import torch
from torchvision import transforms
from src.model import UNet
from src.loss import FocalLoss
from config import config
from src.utils import *
from src.Dataset import *
from src.IoU import *
from src.eval import *
import os


CONFIG = config()
path = CONFIG.path
batch = CONFIG.batch
input_size = config.input_size
load_model_pth = CONFIG.load_model
device = CONFIG.device

# Directory where the results will be saved
result_dir = 'results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if __name__ == "__main__":
    test_transform = transforms.Compose([transforms.Resize(input_size, 0)])

    #to evaluate for all images, change 'test/' to 'all' and  'test_labels' to 'all_labels'
    #pass transform here-in
    test_data = Test(img_pth = path + 'test/', mask_pth = path + 'test_labels', transform=test_transform) 

    #data loaders
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False)

    model = UNet(3, 6, True).to(device)
    criterion = FocalLoss()

    Test_eval(model, testloader, criterion, load_model_pth, device, result_dir)


