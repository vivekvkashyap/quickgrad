from ._module import Module
from ..core import vector
import numpy as np


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def forward(self, y, y_pred):
        loss = ((y - y_pred) ** 2).sum()
        batch = y.shape()[0]
        if self.reduction=='mean':
            loss = loss / batch
        return loss
        
# #add clip and check the implementation properly
# class BCELoss(Module):
#     def __init__(self, reduction='mean', eps = 1e-7):
#         self.reduction = reduction
#         self.eps = eps

#     def forward(self, y, y_pred):
#         first_term = -1 * y * y_pred.log() 
#         second_term =  -1 * (1- y) * (1 - y_pred).log()
#         return (first_term + second_term).sum() / y_pred.shape()[0]
    
#     # def forward(self, y_pred, y):
#     #     # Clip predictions to prevent log(0) or log(1)
#     #     y_pred_clipped = vector(np.clip(y_pred.data, self.eps, 1 - self.eps))
        
#     #     first_term = -1 * y * y_pred_clipped.log()
#     #     second_term = -1 * (1 - y) * (1 - y_pred_clipped).log()
        
#     #     return (first_term + second_term).sum() / y_pred.shape()[0]

class BCELoss(Module):
    def __init__(self, reduction='mean', eps=1e-7):
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, y_pred, y):
        # Ensure numerical stability
        eps_vec = vector(self.eps)
        one_vec = vector(1.0)
        
        # Clamp predictions: max(eps, min(1-eps, y_pred))
        y_pred_safe = y_pred.clamp(self.eps, 1.0 - self.eps)  # You need to implement clamp
        
        # Or manually:
        # y_pred_safe = vector(np.clip(y_pred.data, self.eps, 1 - self.eps))
        
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        pos_loss = y * y_pred_safe.log()
        neg_loss = (one_vec - y) * (one_vec - y_pred_safe).log()
        
        bce = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return bce.mean()  # Use mean() method instead of manual division
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce