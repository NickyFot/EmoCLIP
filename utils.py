import random
import torch
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def PCC(a: torch.tensor, b: torch.tensor):
    am = torch.mean(a, dim=0)
    bm = torch.mean(b, dim=0)
    num = torch.sum((a - am) * (b - bm), dim=0)
    den = torch.sqrt(sum((a - am) ** 2) * sum((b - bm) ** 2)) + 1e-5
    return num/den


def CCC(a: torch.tensor, b: torch.tensor):
    rho = 2 * PCC(a,b) * a.std(dim=0, unbiased=False) * b.std(dim=0, unbiased=False)
    rho /= (a.var(dim=0, unbiased=False) + b.var(dim=0, unbiased=False) + torch.pow(a.mean(dim=0) - b.mean(dim=0), 2) + 1e-5)
    return rho


def feat_scatter(data, labels, class_names):
    color = ['lightcoral', 'darkorange', 'olive', 'teal', 'violet', 'skyblue', 'magenta', 'indigo', 'cyan', 'slategray',
             'lawngreen']
    tsne = TSNE(n_components=2, perplexity=50, n_iter=250)
    results = tsne.fit_transform(data)
    fig, ax = plt.subplots()
    for i in range(len(class_names)):
        mask = labels == i
        mask = mask.reshape(-1)
        ax.scatter(x=results[mask, 0], y=results[mask, 1], color=color[i], label=class_names[i])
    ax.legend(title='class_name')
    return plt


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, output, target):
        mse = F.mse_loss(output, target, reduction='mean')
        return torch.sqrt(mse)


def eval_metrics(y_hat, y):
    mae = torch.abs(y_hat - y)
    mse = torch.pow(mae, exponent=2).mean(0)
    rmse = torch.sqrt(mse)
    return mae.mean(0).cpu().numpy(), mse.cpu().numpy(), rmse.cpu().numpy()
