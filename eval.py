import os
import sys

from sklearn import metrics

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.cuda.amp import autocast

import clip

from config import get_config
from DataLoaders import *
from architecture import VClip

torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"

cnf = get_config(sys.argv)


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    class_tokens = clip.tokenize(CLASS_DESCRIPTION, context_length=77, truncate=True)
    with autocast():
        war = 0
        all_labels = torch.zeros(len(loader.sampler)).to(device)
        all_predictions = torch.zeros(len(loader.sampler)).to(device)
        all_labels_lst = [all_labels.clone().detach() for i in range(int(os.environ['WORLD_SIZE']))]
        all_predictions_lst = [all_predictions.clone().detach() for i in range(int(os.environ['WORLD_SIZE']))]

        for batch_idx, (inputs, labels, _) in enumerate(loader):
            inputs, class_tokens, labels = inputs.to(device), class_tokens.to(device), labels.to(device)
            logits_per_image, logits_per_text = model(inputs, class_tokens)

            predicted = logits_per_image.softmax(dim=-1).argmax(dim=-1, keepdim=True)
            war += predicted.eq(labels.view_as(predicted)).sum().item()

            start_idx = loader.batch_size * batch_idx
            end_idx = start_idx + loader.batch_size
            end_idx = end_idx if end_idx <= all_labels.shape[0] else all_labels.shape[0]

            all_labels[start_idx:end_idx] = labels.reshape(-1)
            all_predictions[start_idx:end_idx] = predicted.reshape(-1)

    dist.all_gather(all_labels_lst, all_labels)
    dist.all_gather(all_predictions_lst, all_predictions)
    return torch.tensor(war, device=device, dtype=torch.float), (torch.cat(all_labels_lst).cpu().numpy(), torch.cat(all_predictions_lst).cpu().numpy())


if __name__ == "__main__":
    # region ddp
    torch.cuda.set_device(cnf.local_rank)
    cnf.is_master = cnf.local_rank == 0
    cnf.device = torch.cuda.device(cnf.local_rank)
    cnf.world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl')
    # endregion

    # region set_up
    if cnf.is_master:
        cnf_dict = vars(cnf)
    # endregion

    model = VClip(num_layers=2)

    if cnf.pretrained:
        state_pth = os.path.join(cnf.pretrained)
        state_dict = torch.load(state_pth)
        new_dict = dict()
        for key in state_dict:
            new_key = key.replace('module.', '')
            new_dict[new_key] = state_dict[key]
        keys = model.load_state_dict(new_dict, strict=False)
        print(keys)

    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[cnf.local_rank],
        output_device=cnf.local_rank
    )
    _, test_loader = get_loaders(cnf)
    war, cm = evaluate(loader=test_loader, model=model)
    dist.all_reduce(war)
    war /= len(test_loader.dataset)
    if cnf.is_master:
        gt, pd = cm[0], cm[1]
        uar = metrics.confusion_matrix(gt, pd, normalize="true").diagonal().mean()
        print('WAR: {}'.format(war*100))
        print('UAR: {}'.format(uar*100))
    dist.destroy_process_group()
