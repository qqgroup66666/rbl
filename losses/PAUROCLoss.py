import torch
import torch.nn as nn

'''
This file includes the pytorch implementation of partial AUC optimization, including one-way and two-way partial AUC.

Note that, this code is based on our follwoing research:

"Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. 
    When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC. ICML-2021. (Long talk)
", and

"Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. 
    Optimizing Two-way Partial AUC with an End-to-end Framework. T-PAMI'2022.

"

'''

class SquareAUCLoss(nn.Module):
    def __init__(self, 
                gamma=1,
                E_k=0,
                weight_scheme='Poly',
                num_class=2,
                reduction='mean',
                eps=1e-6,
                **kwargs):
        super(SquareAUCLoss, self).__init__()

        '''
        args:
            gamma: safe margin in square loss (default = 1.0)
            E_k: warm-up epoch (default = 0), when epoch > E_k, the partial AUC will be conducted.
            weight_scheme: weight scheme, 'Poly' or 'Exp' 
            num_class: only support binary classification
            reduction: loss aggregated manner (default = 'mean')
            eps: use to avoid zero gradient, users can ignore this
        '''

        self.gamma = gamma
        self.reduction = 'mean'

        if num_class != 2:
            raise ValueError("The current version only supports binary classification!")

        self.num_class = num_class
        self.reduction = reduction

        # adopt re_weight func after E_k epoch....
        self.E_k = E_k

        if weight_scheme is not None and weight_scheme not in ['Poly', 'Exp']:
            raise ValueError("weight_scheme should range in [Poly, Exp]")

        self.weight_scheme = weight_scheme

        self.eps = eps

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def forward(self, pred, target, epoch=0):
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        n_plus, n_minus = len(pred_p), len(pred_n)

        weight = self.re_weight(pred_p, pred_n, epoch)
        if pred.is_cuda and not weight.is_cuda:
            weight = weight.cuda()

        pred_p = pred_p.unsqueeze(1).expand(n_plus, n_minus)
        pred_n = torch.reshape(pred_n, (1, n_minus))

        loss = weight * (1.0 + pred_n - pred_p) ** 2
        
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def re_weight(self, pred_p, pred_n, epoch=0):
        return torch.ones(pred_p.shape[0], pred_n.shape[0])

class OPAUCLoss(SquareAUCLoss):
    def __init__(self, gamma, 
                 E_k,
                 weight_scheme,
                 num_class = 2, 
                 reduction='mean', **kwargs):
        
        super(OPAUCLoss, self).__init__(gamma, 
                                        E_k,
                                        weight_scheme,
                                        num_class,
                                        reduction,
                                        **kwargs)

    def re_weight(self, pred_p, pred_n, epoch=0):
        '''
        return:
            must be the (len(pred_p), len(pred_n)) matrix 
                    for element-wise multiplication
        '''

        if epoch < self.E_k:
            return torch.ones(pred_p.shape[0], pred_n.shape[0])

        if self.weight_scheme not in ['Poly', 'Exp']:
            raise ValueError('weight_scheme 4 TPAUC must be included in [Ploy, Exp], but weight_scheme %s' % self.weight_scheme)
        
        if self.weight_scheme == 'Poly':
            col_pred_p = torch.ones_like(pred_p)
            row_pred_n = torch.pow(pred_n + self.eps, self.gamma)
        else:
            col_pred_p = torch.ones_like(pred_p)
            row_pred_n = 1 - torch.exp(- self.gamma * pred_n)

        return torch.mm(col_pred_p.unsqueeze_(1), row_pred_n.unsqueeze_(0))

class TPAUCLoss(SquareAUCLoss):
    def __init__(self, 
                 gamma=0.01, 
                 E_k=10,
                 weight_scheme='Poly',
                 num_class = 2, 
                 reduction='mean', 
                 **kwargs):
        
        super(TPAUCLoss, self).__init__(gamma, 
                                        E_k,
                                        weight_scheme,
                                        num_class,
                                        reduction,
                                        **kwargs)
    def re_weight(self, pred_p, pred_n, epoch=0):
        '''
        return:
            must be the (len(pred_p), len(pred_n)) matrix 
                    for element-wise multiplication
        '''

        if epoch < self.E_k:
            return torch.ones(pred_p.shape[0], pred_n.shape[0])
        
        if self.weight_scheme not in ['Poly', 'Exp']:
            raise ValueError('weight_scheme 4 TPAUC must be included in [Ploy, Exp], but weight_scheme %s' % self.weight_scheme)
        
        if self.weight_scheme == 'Poly':    
            col_pred_p = torch.pow((1 - pred_p + self.eps), self.gamma)
            row_pred_n = torch.pow(pred_n + self.eps, self.gamma)
        elif self.weight_scheme == 'Exp':
            col_pred_p = 1 - torch.exp(- self.gamma * (1 - pred_p))
            row_pred_n = 1 - torch.exp(- self.gamma * pred_n)

        return torch.mm(col_pred_p.unsqueeze_(1), row_pred_n.unsqueeze_(0))

class MinMaxTPAUC(SquareAUCLoss):
    def __init__(self, 
                gamma=1,
                E_k=0,
                weight_scheme=None,
                num_class=2,
                eps=1e-6,
                first_state_loss=None,
                reg_a=0.0,
                reg_b=0.0,
                **kwargs):
        
        super(MinMaxTPAUC, self).__init__()

        '''
        args:
            reg_a and reg_b: weight of the strong convex constraint
            first_state_loss: warmup loss (default = None), could be 'SquareAUCLoss' or other pytorch supported loss such as CE.
        '''

        self.a = torch.zeros(10, dtype=torch.float64, device="cuda", requires_grad=True)
        self.b = torch.zeros(8, dtype=torch.float64, device="cuda", requires_grad=True)

        assert self.a.requires_grad == True
        assert self.b.requires_grad == True

        self.gamma = gamma
        self.E_k = E_k

        if weight_scheme is not None and weight_scheme not in ['Poly', 'Exp']:
            raise ValueError
        self.weight_scheme = weight_scheme

        self.num_class = num_class

        self.first_stage_loss = first_state_loss() 
        self.eps = eps

        self.reg_a = reg_a
        self.reg_b = reg_b

        if kwargs is not None:
            self.__dict__.update(kwargs)
        
    def function_w(self, x, y):

        assert x.shape == y.shape, "dim of x and y must be the same!"

        return 2 * x * y - torch.square(x)
    
    def function_v(self, b, a1, a2, e, f):
        return self.function_w(b, e + f) - self.function_w(a1, e) - self.function_w(a2, f)

    def forward(self, pred, target, epoch=0):

        if self.first_stage_loss is not None and epoch < self.E_k:
            return self.first_stage_loss(pred, target, epoch)
        
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        
        if self.weight_scheme == 'Poly':
            v_plus = torch.pow(1 - pred_p + self.eps, self.gamma)
            v_minus = torch.pow(pred_n + self.eps, self.gamma)
        else:
            v_plus = 1 - torch.exp(- self.gamma * (1 - pred_p))
            v_minus = 1 - torch.exp(- self.gamma * pred_n)
        
        c_plus = v_plus.mean()
        c_minus = v_minus.mean()

        f_plus = (v_plus * pred_p).mean()
        f_minus = (v_minus * pred_n).mean()

        f_plus_sq = (v_plus * pred_p.square()).mean()
        f_minus_sq = (v_minus * pred_n.square()).mean()

        loss = 0.5 * self.function_v(self.b[0], self.a[0], self.a[1], c_plus, c_minus) - \
             self.function_v(self.a[2], self.b[1], self.b[2], c_minus, f_plus) + \
             self.function_v(self.b[3], self.a[3], self.a[4], c_plus, f_minus) + \
             0.5 * self.function_v(self.b[4], self.a[5], self.a[6], c_minus, f_plus_sq) + \
             0.5 * self.function_v(self.b[5], self.a[7], self.a[8], c_plus, f_minus_sq) - \
             self.function_v(self.a[9], self.b[6], self.b[7], f_plus, f_minus) 

        return loss.mean() + self.strong_convex_loss()
        
    def strong_convex_loss(self):
        return self.reg_a * self.a.square().sum() - self.reg_b * self.b.square().sum()