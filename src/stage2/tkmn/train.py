import sys
import copy
from pathlib import Path
import argparse
from collections import OrderedDict

import pandas as pd

import torch
from torch.utils.data import DataLoader

import yaml
from tqdm import tqdm

import source.datasets as datasets
from source.datasets import build_transforms, DatasetPhase
import source.models as models
from source.metrics import RSNA2024Metrics
from source.submit import Submit

from source.utils.seed import fix_seed
from source.utils.config import get_config

from src.utils import load_settings


def aggregate_submission_csv(dst_root: Path):
    pred_df_list = []
    for p in dst_root.glob('**/exp*best_score_submission.csv'):
        pred_df_list.append(pd.read_csv(p))
    submit_df = pd.concat(pred_df_list).reset_index(drop=True)
    submit_df.to_csv(dst_root / 'best_score_submission.csv', index=False)


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    autocast: torch.cuda.amp.autocast,
    scaler: torch.cuda.amp.GradScaler,
    use_tqdm: bool
):
    model.train()
    total_loss = 0.0
    dataloader = tqdm(train_dataloader, leave=True) if use_tqdm else train_dataloader
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_ids, segment_ids, condition_ids, available_flags = batch
        if torch.all(~available_flags):
            # データセット作成時、各レベルのキーポイントを正しく検出できない場合はレベル分割画像を作成しない。
            # 分割画像を用意できなかったstudyはtargetに-100を設定することでloss計算から除外し、さらにavailable_flagsにFalseを設定する。
            # Batch内のtargetがすべて-100の場合、loss計算するとnanになってしまうので学習処理自体をスキップする。
            if use_tqdm:
                dataloader.set_postfix(OrderedDict(loss='-', lr=f'{optimizer.param_groups[0]["lr"]:.3e}'))
            if scheduler is not None:
                scheduler.step()
            continue

        sagittal_t1_image = sagittal_t1_image.to(device)
        sagittal_t2_image = sagittal_t2_image.to(device)
        axial_t2_image = axial_t2_image.to(device)
        label = label.to(device)

        with autocast:
            outputs = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image, label)
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

        if use_tqdm:
            dataloader.set_postfix(
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
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    device: torch.device,
    autocast: torch.cuda.amp.autocast,
    metrics: RSNA2024Metrics,
    submit: Submit,
    use_tqdm: bool
):
    model.eval()

    total_loss = 0
    logits = []
    labels = []
    study_ids_list = []
    segment_ids_list = []
    condition_ids_list = []
    available_flags_list = []

    dataloader = tqdm(valid_dataloader, leave=True) if use_tqdm else valid_dataloader
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, study_ids, segment_ids, condition_ids, available_flags = batch
            if torch.all(~available_flags):
                # データセット作成時、各レベルキーポイントを正しく検出できない場合はレベル分割画像を作成しない。
                # 分割画像を用意できなかったstudyはtargetに-100を設定することでloss計算から除外し、さらにavailable_flagsにFalseを設定する。
                # Batch内のtargetがすべて-100の場合、loss計算するとnanになってしまうので学習処理自体をスキップする。
                continue

            sagittal_t1_image = sagittal_t1_image.to(device)
            sagittal_t2_image = sagittal_t2_image.to(device)
            axial_t2_image = axial_t2_image.to(device)
            label = label.to(device)

            with autocast:
                outputs = model(sagittal_t1_image, sagittal_t2_image, axial_t2_image, label, force_loss_execute=True)
                total_loss += outputs['losses']['loss'].item()
                logits.append(outputs['logits'].detach().cpu())
                labels.append(label.detach().cpu())
                study_ids_list.append(study_ids)
                segment_ids_list.append(segment_ids)
                condition_ids_list.append(condition_ids)
                available_flags_list.append(available_flags)

    val_loss = total_loss / len(valid_dataloader)

    logits = torch.cat(logits, dim=0)
    preds = torch.softmax(logits.float(), dim=-1)
    logits = logits.numpy()
    preds = preds.numpy()
    targets = torch.cat(labels).numpy()
    available_flags = torch.cat(available_flags_list).numpy()

    study_ids = torch.cat(study_ids_list).numpy()
    segment_ids = torch.cat(segment_ids_list).numpy()
    condition_ids = torch.cat(condition_ids_list).numpy()
    submit_df = submit(preds, study_ids, segment_ids, condition_ids, available_flags)
    val_score = metrics(submit_df)

    conditions = list(set([row_id.split('_', 1)[1].rsplit('_', 2)[0] for row_id in submit_df['row_id']]))
    return dict(val_loss=val_loss, val_score=val_score, conditions=conditions, submit_df=submit_df)


def perform_cross_validation(dst_root: Path, metrics: RSNA2024Metrics):
    pred_df_list = []
    for p in dst_root.glob('**/*best_score_submission.csv'):
        pred_df_list.append(pd.read_csv(p))
    cv_df = pd.concat(pred_df_list).reset_index(drop=True)
    cv_score = metrics(cv_df)
    print(cv_score)
    return cv_score


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
    config['dataset']['image_csv_path'] = str(settings.train_data_clean_dir / config['dataset']['image_csv_path'])
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
    label_ori_df = pd.read_csv(label_csv_path)
    label_df = label_ori_df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    label_df = label_df.replace(label2id)

    image_csv_path = config['dataset'].pop('image_csv_path')
    image_df = pd.read_csv(image_csv_path)

    # metrics
    metrics = RSNA2024Metrics(label_ori_df)

    # submit
    submit = Submit(**config['submit'])

    # debug
    if config.get('debug', {}).get('use_debug', False):
        debug_mode = True
        config['epochs'] = config['debug'].get('epochs', 2)
        config['dataloader']['batch_size'] = config['debug'].get('batch_size', 2)
        config['dataloader']['num_workers'] = config['debug'].get('num_workers', 4)

        # hack for debug
        def dummy_metrics(self, submit_df):
            return 0.0
        RSNA2024Metrics.__call__ = dummy_metrics
    else:
        debug_mode = False

    base_config = copy.deepcopy(config)
    best_metrics = []
    used_folds = config['folds']
    for fold in used_folds:
        print(f'fold: {fold}')
        config = copy.deepcopy(base_config)

        # get fold
        fold_train_df = label_df[label_df['fold'] != fold].reset_index(drop=True).copy()
        fold_valid_df = label_df[label_df['fold'] == fold].reset_index(drop=True).copy()

        # debug
        if debug_mode:
            subset_size = config['debug'].get('subset_size', 12)
            fold_train_df = fold_train_df.sample(n=subset_size, random_state=config['seed']).reset_index(drop=True)

        # transorm
        train_transforms = {}
        valid_transforms = {}
        for plane in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
            if plane in config['transform']:
                train_transforms[plane] = build_transforms(DatasetPhase.TRAIN, **config['transform'][plane])
                valid_transforms[plane] = build_transforms(DatasetPhase.VALIDATION, **config['transform'][plane])

        # data loader
        dataset_name = config['dataset'].pop('name')
        dataset_class = getattr(datasets, dataset_name)
        train_dataset = dataset_class(**config['dataset'],
                                      train_df=fold_train_df,
                                      train_image_df=image_df,
                                      phase=DatasetPhase.TRAIN,
                                      transforms=train_transforms)
        valid_dataset = dataset_class(**config['dataset'],
                                      train_df=fold_valid_df,
                                      train_image_df=image_df,
                                      phase=DatasetPhase.VALIDATION,
                                      transforms=valid_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'],
                                      shuffle=True, pin_memory=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config['dataloader']['batch_size'], num_workers=config['dataloader']['num_workers'],
                                      shuffle=False, pin_memory=True, drop_last=False)

        # model
        model_class = getattr(models, config['model'].pop('name'))
        model = model_class(**config['model'])
        model.train()
        model.to(device)

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

        # initialize training variables
        start_epoch = 1
        best_val_score = float('inf')
        best_val_loss = float('inf')
        conditions = None
        log_data = []

        # resume fron checkpoint
        if config.get('resume', False):
            checkpoint_path = args.dst_root / f'exp{experiment_name}_fold{fold}_latest.pth'
            log_path = args.dst_root / f'exp{experiment_name}_fold{fold}_log.csv'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_score = checkpoint['best_metrics']['val_score']
                best_val_loss = checkpoint['best_metrics']['val_loss']
                conditions = checkpoint['conditions']
                print(f"Resuming from epoch {start_epoch} for fold {fold}")
                if log_path:
                    log_data = pd.read_csv(log_path).to_dict('records')

        # training
        for epoch in range(start_epoch, config['epochs'] + 1):
            train_metrics = train_one_epoch(model, train_dataloader, optimizer, scheduler, device, autocast, scaler, config['use_tqdm'])
            valid_metrics = validation_one_epoch(model, valid_dataloader, device, autocast, metrics, submit, config['use_tqdm'])
            submit_df = valid_metrics.pop('submit_df')
            conditions = valid_metrics.pop('conditions')

            print(f'epoch {epoch} train_loss: {train_metrics["train_loss"]}, val_loss: {valid_metrics["val_loss"]}, val_score: {valid_metrics["val_score"]}')
            lr = optimizer.param_groups[0]["lr"]
            log_data.append(dict(epoch=epoch, lr=lr, **train_metrics, **valid_metrics))

            if valid_metrics['val_loss'] < best_val_loss:
                best_val_loss = valid_metrics['val_loss']
                submit_df.to_csv(args.dst_root / f'exp{experiment_name}_fold{fold}_best_loss_submission.csv', index=False)
                torch.save(model.state_dict(), args.dst_root / f'exp{experiment_name}_fold{fold}_best_loss.pth')
            if valid_metrics['val_score'] < best_val_score:
                best_val_score = valid_metrics['val_score']
                submit_df.to_csv(args.dst_root / f'exp{experiment_name}_fold{fold}_best_score_submission.csv', index=False)
                torch.save(model.state_dict(), args.dst_root / f'exp{experiment_name}_fold{fold}_best_score.pth')

            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metrics': dict(val_loss=best_val_loss, val_score=best_val_score),
                'fold': fold,
                'conditions': conditions,
            }, args.dst_root / f'exp{experiment_name}_fold{fold}_latest.pth')
            submit_df.to_csv(args.dst_root / f'exp{experiment_name}_fold{fold}_latest_submission.csv', index=False)

            # 経過を観察したいので毎エポック出力する
            pd.DataFrame(log_data).to_csv(args.dst_root / f'exp{experiment_name}_fold{fold}_log.csv', index=False)

        best_metrics.append(dict(fold=fold, best_val_loss=best_val_loss, best_val_score=best_val_score, conditions=conditions))

    pd.DataFrame(best_metrics).to_csv(args.dst_root / f'exp{experiment_name}_best_metrics.csv', index=False)
    aggregate_submission_csv(args.dst_root)
    perform_cross_validation(args.dst_root, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dst_root', type=str, default='./outputs/train/')
    parser.add_argument('--options', type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f'config: {args.config}')
    print(f'experiment: {args.experiment}')
    print(f'dst_root: {args.dst_root}')
    print(f'options: {args.options}')
    main(args)
