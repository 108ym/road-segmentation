import torch
import torch.nn as nn
import numpy as np
from src.IoU import *
from src.utils import *
from config import config
import os
from PIL import Image

CONFIG = config()

def Validate(model, validloader, criterion, valid_loss_min, device, model_path):
    valid_loss = 0
    val_iou = []
    val_losses = []
    model.eval()
    for i, val_data in enumerate(validloader):
        inp, masks, _ = val_data
        inp, masks = inp.float(), masks.float()  # Ensures tensors are float32
        inp, masks = inp.to(device), masks.to(device)
        out = model(inp)
        val_target = masks.argmax(1)
        val_loss = criterion(out, val_target.long())
        valid_loss += val_loss.item() * inp.size(0)
        iou = iou_pytorch(out.argmax(1), val_target)
        val_iou.extend(iou)    
    miou = torch.FloatTensor(val_iou).mean()
    valid_loss = valid_loss / len(validloader.dataset)
    val_losses.append(valid_loss)
    print(f'\t\t Validation Loss: {valid_loss:.4f},',f' Validation mIoU: {miou:.3f}')
    
    if np.mean(val_losses) <= valid_loss_min:
        torch.save(model.state_dict(), model_path+'/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,np.mean(val_losses))+'\n')
        valid_loss_min = np.mean(val_losses)

    return valid_loss, valid_loss_min


def Test_eval(model, testloader, criterion, model_save_pth, device, result_dir):
    model.load_state_dict(torch.load(model_save_pth))
    model.eval()
    test_loss = 0
    test_iou = []
    imgs, masks, preds = [], [], []
    with torch.no_grad():
        for i, test_data in enumerate(testloader):
            inp, mask, _ = test_data
            inp, mask = inp.float(), mask.float()  # Ensures tensors are float32
            inp, mask = inp.to(device), mask.to(device)
            
            out = model(inp)        
            target = mask.argmax(1)
            loss = criterion(out, target.long())
            test_loss += loss.item() * inp.size(0)
            iou = iou_pytorch(out.argmax(1), target)
            test_iou.extend(iou)  
            test_miou = torch.FloatTensor(test_iou).mean()
  
            # Convert model output to RGB mask and save
            pred_rgb = mask_to_rgb(out.detach().cpu(), CONFIG.id2code)
            pred_image = pred_rgb[0] 
            pred_image = pred_image.astype(np.uint8) 
            img = Image.fromarray(pred_image)
            img.save(os.path.join(result_dir, f'predicted_mask_{i}.png'))

    test_loss /= len(testloader.dataset)
    print(f"Test loss is: {test_loss:.4f}")
    print(f'Test mIoU: {test_miou:.3f}')

