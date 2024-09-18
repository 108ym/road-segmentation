import torch
import torch.nn as nn



ALPHA = torch.tensor([1.2, 1.2, 1.2, 1.2, 3.0, 3.1]) 

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=ALPHA):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        alpha = self.alpha.to(inputs.device)
        alpha_t = alpha[targets]

        weighted_focal_loss = alpha_t * focal_loss
        return weighted_focal_loss.mean()

