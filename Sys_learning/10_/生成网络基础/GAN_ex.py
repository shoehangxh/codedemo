import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.linear import Linear
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


n_idea = 10
art_f = 200
lr_D = 0.0001
lr_G = 0.0001
batch_size = 64
PAINT_POINTS = np.vstack([np.linspace(-1, 1, art_f) for _ in range(batch_size)])

G = nn.Sequential(
    nn.Linear(n_idea,128),
    nn.ReLU(inplace = True),
    nn.Linear(128,art_f)
)
D = nn.Sequential(
    nn.Linear(art_f,128),
    nn.ReLU(inplace = True),
    nn.Linear(128,1),
    nn.Sigmoid()
)

opt_D = torch.optim.Adam(D.parameters(),lr = lr_D)
opt_G = torch.optim.Adam(G.parameters(),lr = lr_G)

def artist_works():     
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]#(64,)-->(64,1)
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)

for step in range(1000):
    r_paints = artist_works()
    g_seed = torch.randn(size = (batch_size,n_idea),requires_grad=True)
    g_paints = G(g_seed)

    to_up = D(r_paints)#0
    to_down = D(g_paints)#1

    G_loss = torch.mean(torch.log(1. - to_down))  
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    to_down = D(g_paints.detach())  # D try to reduce this prob，切断反向传播，提出一个新的tensor不在计算图中
    D_loss = - torch.mean(torch.log(to_up) + torch.log(1. - to_down))      

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    if step % 1000 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], g_paints.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % to_up.data.numpy().mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=12);plt.draw();plt.pause(0.01)
        plt.show()
