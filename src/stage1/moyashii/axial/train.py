import sys
import copy
from pathlib import Path
import argparse
from collections import OrderedDict

import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml
from tqdm import tqdm

from source.datasets import RSNA2024KeypointDatasetTrain, build_transforms, DatasetPhase
from source.models import RSNA2024KeypointNet
from source.metrics import RSNA2024KeypointMetrics

from source.utils.seed import fix_seed
from source.utils.config import get_config

from src.utils import load_settings


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    autocast: torch.cuda.amp.autocast,
    scaler: torch.cuda.amp.GradScaler
):
    model.train()
    total_loss = 0.0
    with tqdm(train_dataloader, leave=True) as pbar:
        optimizer.zero_grad()
        for idx, batch in enumerate(pbar):
            images, targets, study_id, keypoints = batch
            images = images.to(device)
            targets = targets.to(device)

            with autocast:
                outputs = model(images, targets)
                loss = outputs['losses']['loss']
                total_loss += loss.item()

            if not torch.isfinite(loss):
                print(f"Loss is {loss}, stopping training")
                sys.exit(1)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            pbar.set_postfix(
                OrderedDict(
                    loss=f'{loss.item():.6f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                )
            )

            if scheduler is not None:
                scheduler.step()

    train_loss = total_loss / len(train_dataloader)

    return dict(train_loss=train_loss)


def validation_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    device: torch.device,
    autocast: torch.cuda.amp.autocast,
    metrics: RSNA2024KeypointMetrics,
):
    model.eval()

    total_loss = 0
    study_ids = []
    preds = []
    keypoints_list = []
    # DEBUG!!!!
    images_list = []

    with tqdm(valid_dataloader, leave=True) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                images, targets, study_id, keypoints = batch
                images = images.to(device)
                targets = targets.to(device)

                with autocast:
                    outputs = model(images, targets, force_loss_execute=True)
                    total_loss += outputs['losses']['loss'].item()
                    preds.append(outputs['logits'].detach().sigmoid().cpu())
                    keypoints_list.append(keypoints.detach().cpu())
                    study_ids.append(study_id)

                    # DEBUG!!!!
                    images_list.append(images.detach().cpu())

    val_loss = total_loss / len(valid_dataloader)
    preds = torch.cat(preds, dim=0).numpy()
    study_ids = torch.cat(study_ids).numpy()
    keypoints = torch.cat(keypoints_list).numpy()

    metrics_results = metrics(keypoints, preds)
    val_score = metrics_results.pop('score')
    results = dict(val_loss=val_loss, val_score=val_score)
    results.update(metrics_results)

    return results


def main(args: argparse.Namespace):
    settings = load_settings()

    # output directory
    args.dst_root = settings.model_checkpoint_dir / args.dst_root
    args.dst_root.mkdir(parents=True, exist_ok=True)
    print(f'dst_root: {args.dst_root}')

    # config
    config = get_config(args.config, args.options)
    config['dataset']['image_root'] = str(settings.train_data_clean_dir / config['dataset']['image_root'])
    config['dataset']['label_csv_path'] = str(settings.train_data_clean_dir / config['dataset']['label_csv_path'])
    print(yaml.dump(config))

    # seed
    fix_seed(config['seed'])

    # device
    device = torch.device(config['device'])

    # experiment name
    experiment_name = args.experiment
    print(f'experiment_name: {experiment_name}')

    autocast = torch.cuda.amp.autocast(enabled=config['use_amp'], dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'], init_scale=4096)

    # read csv
    label_csv_path = config['dataset'].pop('label_csv_path')
    label_df = pd.read_csv(label_csv_path)

    # metrics
    metrics = RSNA2024KeypointMetrics(**config['metrics'])

    base_config = copy.deepcopy(config)
    best_metrics = []
    for fold in config['folds']:
        print(f'fold: {fold}')
        config = copy.deepcopy(base_config)

        # get fold
        fold_train_df = label_df[label_df['fold'] != fold].reset_index(drop=True).copy()
        fold_valid_df = label_df[label_df['fold'] == fold].reset_index(drop=True).copy()

        # transorm
        train_transform = build_transforms(DatasetPhase.TRAIN, **config['transform'])
        valid_transform = build_transforms(DatasetPhase.VALIDATION, **config['transform'])

        # data loader
        train_dataset = RSNA2024KeypointDatasetTrain(**config['dataset'],
                                                     train_df=fold_train_df,
                                                     phase=DatasetPhase.TRAIN,
                                                     transform=train_transform,)
        valid_dataset = RSNA2024KeypointDatasetTrain(**config['dataset'],
                                                     train_df=fold_valid_df,
                                                     phase=DatasetPhase.VALIDATION,
                                                     transform=valid_transform,)
        train_dataloader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'],
                                      shuffle=True, pin_memory=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'],
                                      shuffle=False, pin_memory=True, drop_last=False)

        # model
        model = RSNA2024KeypointNet(**config['model'])
        model.train()
        model.to(device)
        if 'pretrained' in config:
            model.load_state_dict(torch.load(config['pretrained'], map_location=device))

        # optimizer and scheduler
        optimizer_name = config['optimizer'].pop('name')
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), **config['optimizer'])
        scheculer_name = config['scheduler'].pop('name')
        if scheculer_name == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            **config['scheduler'],
                                                            epochs=config['epochs'],
                                                            steps_per_epoch=len(train_dataloader))
        else:
            raise ValueError(f'{scheculer_name} is not supported.')

        # training
        log_data = []
        best_val_loss = float('inf')
        best_val_score = -float('inf')
        for epoch in range(1, config['epochs']+1):
            train_metrics = train_one_epoch(epoch, model, train_dataloader, optimizer, scheduler, device, autocast, scaler)
            valid_metrics = validation_one_epoch(epoch, model, valid_dataloader, device, autocast, metrics)

            metrics_str = f'epoch: {epoch}, '
            metrics_str += ', '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])
            metrics_str += ', '
            metrics_str += ', '.join([f'{k}: {v:.4f}' for k, v in valid_metrics.items()])
            print(metrics_str)
            lr = optimizer.param_groups[0]["lr"]
            log_data.append(dict(epoch=epoch, lr=lr, **train_metrics, **valid_metrics))

            if valid_metrics['val_loss'] < best_val_loss:
                best_val_loss = valid_metrics['val_loss']
                torch.save(model.state_dict(), args.dst_root / f'exp{experiment_name}_fold{fold}_best_loss.pth')
            if valid_metrics['val_score'] > best_val_score:
                best_val_score = valid_metrics['val_score']
                torch.save(model.state_dict(), args.dst_root / f'exp{experiment_name}_fold{fold}_best_score.pth')
            torch.save(model.state_dict(), args.dst_root / f'exp{experiment_name}_fold{fold}_latest.pth')

            # 経過を観察したいので毎エポック出力する
            pd.DataFrame(log_data).to_csv(args.dst_root / f'exp{experiment_name}_fold{fold}_log.csv', index=False)

        best_metrics.append(dict(fold=fold, best_val_loss=best_val_loss, best_val_score=best_val_score))

    pd.DataFrame(best_metrics).to_csv(args.dst_root / f'exp{experiment_name}_best_metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dst_root', type=str, default='axial_kpt',
                        help='Relative path from the directory specified by the MODEL_CHECKPOINT_DIR in the SETTINGS.json.')
    parser.add_argument('--options', type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f'config: {args.config}')
    print(f'experiment: {args.experiment}')
    print(f'dst_root: {args.dst_root}')
    print(f'options: {args.options}')
    main(args)
