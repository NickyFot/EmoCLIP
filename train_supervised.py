import os
import sys

from sklearn import metrics

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch import optim
from torch.optim import lr_scheduler

import clip

import wandb

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


def train(loader, model, loss_criterion, optimizer):
    model.train()
    losses = torch.zeros(len(loader)).to(device)
    for batch_idx, (inputs, labels, _) in enumerate(loader):
        labels = labels.reshape(-1).to(device)
        with autocast():
            optimizer.zero_grad()
            descriptions = [CLASS_DESCRIPTION[i] for i in labels]
            class_tokens = clip.tokenize(descriptions, context_length=77, truncate=True)
            inputs, class_tokens = inputs.to(device), class_tokens.to(device)
            logits_per_image, logits_per_text = model(inputs, class_tokens)
            ground_truth = torch.arange(len(inputs), dtype=torch.long, device=device)

            loss_i = loss_criterion(logits_per_image, ground_truth)
            loss_t = loss_criterion(logits_per_text, ground_truth)
            loss = (loss_i + loss_t) / 2

            losses[batch_idx] = loss.item()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                '\r Iter[{}/{}]\t loss: {:.2f} '.format(
                    batch_idx + 1,
                    len(loader),
                    loss.item()
                )
            )
            sys.stdout.flush()

    return losses.mean()


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
        ROOT_FOLDER = os.path.join(cnf.log_dir, 'checkpoints')
        EXP_FOLDER = os.path.join(ROOT_FOLDER, cnf.exp_name)
        MODELS_FOLDER = os.path.join(EXP_FOLDER, 'models')
        PREDS_FOLDER = os.path.join(EXP_FOLDER, 'preds')
        if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER, exist_ok=True)
        if not os.path.exists(PREDS_FOLDER):
            os.makedirs(PREDS_FOLDER, exist_ok=True)
        cnf_dict = vars(cnf)
    # endregion

    model = VClip(num_layers=2)

    if cnf.pretrained:
        state_pth = os.path.join(cnf.pretrained, '{}.pth'.format(cnf.fold))
        state_dict = torch.load(state_pth)
        new_dict = dict()
        for key in state_dict:
            new_key = key.replace('module.', '')
            new_dict[new_key] = state_dict[key]
        keys = model.load_state_dict(new_dict, strict=False)
        print(keys)

    if cnf.finetune:
        if cnf.text:
            for name, param in model.backbone.transformer.named_parameters():
                param.requires_grad = True
        if cnf.visual:
            for name, param in model.backbone.visual.named_parameters():
                param.requires_grad = True
        backbone_params = model.backbone.parameters()
        other_params = list()
        for name, param in model.named_parameters():
            if 'backbone' not in name:
                other_params.append(param)
        param_groups = [
            {'params': other_params},
            {'params': backbone_params, 'lr': cnf.lr * 0.001}
        ]
    else:
        param_groups = model.parameters()

    if cnf.optim == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=cnf.lr,
            momentum=0.9,
            weight_decay=cnf.wd
        )
    else:
        optimizer = optim.Adam(
            param_groups,
            lr=cnf.lr,
            # betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=cnf.wd
        )
    scaler = GradScaler()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    loss_c = nn.CrossEntropyLoss()

    if cnf.is_master:
        wandb.init(project='ZeroShot', group=cnf.exp_name, notes='', config=cnf_dict, job_type='fold={}'.format(cnf.fold))
        wandb.watch(model, log="all", log_freq=25)
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[cnf.local_rank],
        output_device=cnf.local_rank
    )
    train_loader, test_loader = get_loaders(cnf, fold=cnf.fold)
    w = 0
    for e in range(cnf.num_epochs):
        train_loader.sampler.set_epoch(e)
        test_loader.sampler.set_epoch(e)
        train_loss = train(loader=train_loader, model=model, loss_criterion=loss_c, optimizer=optimizer)
        dist.all_reduce(train_loss)
        train_loss /= cnf.world_size
        war, cm = evaluate(loader=test_loader, model=model)
        dist.all_reduce(war)
        war /= len(test_loader.dataset)
        if cnf.is_master:
            gt, pd = cm[0], cm[1]
            uar = metrics.confusion_matrix(gt, pd, normalize="true").diagonal().mean()
            test_dict = {
                'epoch': e,
                'loss': train_loss,
                'war': war,
                'uar': uar,
                'lr': scheduler.get_last_lr()[0]
            }
            wandb.log(test_dict)
        if war > w:
            # preds = get_pred(test_loader, model)
            w = war
            if cnf.is_master:
                model_filename = os.path.join(MODELS_FOLDER, '{}.pth'.format(cnf.fold))
                torch.save(model.state_dict(), model_filename)
                best = wandb.Table(columns=["WAR", "UAR"], data=[[war, uar]])
                wandb.log({'best_results': best})
                wandb.log({"conf_mat": wandb.plot.confusion_matrix(preds=pd.reshape(-1), y_true=gt.reshape(-1),
                                                                   class_names=CLASSES)})
        scheduler.step()
    dist.destroy_process_group()
