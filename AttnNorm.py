import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttNorm(nn.Module):
    def __init__(self, nc, kama=10, nClass=16, orth_lambda=1e-3, eps=1e-7, affine=False):
        super().__init__()
        self.nc = nc
        self.kama = kama
        self.nClass = nClass
        self.orth_lambda = orth_lambda
        self.eps = eps
        self.affine = affine
        
        self.conv_k = nn.Conv2d(nc, nc//8, 1)
        self.conv_q = nn.Conv2d(nc, nc//8, 1)
        self.conv_v = nn.Conv2d(nc, nc, 1)
        
        self.xmask_filter = nn.Parameter(torch.randn(nClass, nc, 1, 1))
        self.alpha = nn.Parameter(torch.ones(1, nClass, 1, 1))
        self.sigma = nn.Parameter(torch.zeros(1))
        
        if affine:
            print('AttNorm with affine')
            self.gamma = nn.Parameter(torch.ones(1, nc, nClass, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, nc, nClass, 1, 1))
        
    def forward(self, x):
        b, c, h, w = x.size()
        x_mask = F.conv2d(x, self.xmask_filter)
        mask_w = torch.reshape(self.xmask_filter, (self.nClass, self.nc))
        sym = torch.matmul(mask_w, torch.t(mask_w))
        sym -= torch.eye(self.nClass).cuda()
        orth_loss = self.orth_lambda * torch.norm(sym, p='fro')
        
        # sampling _pos
        sampling_pos = torch.multinomial(torch.ones(1, h*w)*0.5, self.nClass)
        sampling_pos = torch.squeeze(sampling_pos, dim=0)
        sampling_pos = sampling_pos.reshape(1, 1, -1).repeat(b, 1, 1).repeat(1, c//8, 1).cuda()
        xk = self.conv_k(x)
        xk_reshaped = xk.reshape((b, c//8, h*w))
        fast_filters = torch.gather(xk_reshaped, dim=2, index=sampling_pos)

        
        xq = self.conv_q(x)
        xq_reshaped = xq.reshape((b, c//8, h*w))
        fast_activations = torch.matmul(fast_filters.permute(0, 2, 1), xq_reshaped)
        fast_activations = fast_activations.reshape((b, self.nClass, h, w))
        alpha = torch.clamp(self.alpha, 0, 1)
        layout = nn.Softmax(dim=1)((alpha * fast_activations + x_mask) / self.kama)

        
        layout_expand = torch.unsqueeze(layout, dim=1)
        cnt = torch.sum(layout_expand, dim=(3,4), keepdim=True) + self.eps
        xv = self.conv_v(x)
        xv_expand = torch.unsqueeze(xv, dim=2).repeat(1,1,self.nClass,1,1)
        hot_area = xv_expand * layout_expand
        xv_mean = torch.mean(hot_area, dim=(3,4), keepdim=True)/cnt
        xv_std = torch.sqrt(torch.sum((hot_area - xv_mean)**2, dim=(3,4), keepdim=True)/cnt)
        if self.affine:
            xn = torch.sum(( ((xv_expand-xv_mean)/(xv_std+self.eps))*self.gamma + self.beta) * layout_expand, dim=2)
        else:
            xn = torch.sum(((xv_expand-xv_mean)/(xv_std+self.eps)) * layout_expand, dim=2)
        x = x + self.sigma * xn

        return x, layout, orth_loss
