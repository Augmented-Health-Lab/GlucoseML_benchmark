# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 04:04:32 2022

@author: tz2916
"""

import torch



class QuantileLoss(object):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as
    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(self,quantiles = [ 0.1, 0.25, 0.5, 0.75, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
 
    
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., [i]]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))

        return loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.
        Args:
            y_pred: prediction output of network
        Returns:
            torch.Tensor: point prediction
        """

        idx = self.quantiles.index(0.5)
        y_pred = y_pred[..., [idx]]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.
        Args:
            y_pred: prediction output of network
        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred
    
    def __call__(self, x, y):
        return self.loss(x, y)
    