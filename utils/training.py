from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from datasets.survival import GenericMILSurvivalDataset, save_splits
from models.registry import build_model
from utils.file_utils import save_pkl
from utils.losses import CoxSurvLoss, CrossEntropySurvLoss, NLLSurvLoss

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:  # pragma: no cover
    concordance_index_censored = None

try:
    from lifelines.utils import concordance_index
except ImportError:  # pragma: no cover
    concordance_index = None


def seed_torch(seed: int) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def survival_c_index(event_times, censorships, risk_scores) -> float:
    event_observed = (1 - np.asarray(censorships)).astype(bool)
    event_times = np.asarray(event_times, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)

    if concordance_index_censored is not None:
        return float(concordance_index_censored(event_observed, event_times, risk_scores, tied_tol=1e-8)[0])
    if concordance_index is not None:
        return float(concordance_index(event_times, -risk_scores, event_observed=event_observed))
    raise ImportError("Install scikit-survival or lifelines to compute the c-index.")


def collate_survival(batch):
    path_features = torch.cat([item[0] for item in batch], dim=0)
    omic_features = torch.stack([item[1] for item in batch]).float()
    labels = torch.LongTensor([item[2] for item in batch])
    event_times = np.array([item[3] for item in batch], dtype=float)
    censorships = torch.FloatTensor([item[4] for item in batch])
    return path_features, omic_features, labels, event_times, censorships


def collate_survival_signatures(batch):
    path_features = torch.cat([item[0] for item in batch], dim=0)
    omics = [torch.stack([item[group_idx] for item in batch]).float() for group_idx in range(1, 7)]
    labels = torch.LongTensor([item[7] for item in batch])
    event_times = np.array([item[8] for item in batch], dtype=float)
    censorships = torch.FloatTensor([item[9] for item in batch])
    return (path_features, *omics, labels, event_times, censorships)


def make_weights_for_balanced_classes(dataset) -> torch.DoubleTensor:
    num_samples = float(len(dataset))
    class_weights = [num_samples / max(len(class_ids), 1) for class_ids in dataset.slide_cls_ids]
    weights = []
    for idx in range(len(dataset)):
        label = dataset.getlabel(idx)
        weights.append(class_weights[label])
    return torch.DoubleTensor(weights)


def get_split_loader(dataset, training=False, weighted=False, mode="coattn", batch_size=1):
    collate_fn = collate_survival_signatures if mode == "coattn" else collate_survival
    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes(dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, **kwargs)


def get_loss(args):
    if args.bag_loss == "nll_surv":
        return NLLSurvLoss(alpha=args.alpha_surv)
    if args.bag_loss == "ce_surv":
        return CrossEntropySurvLoss(alpha=args.alpha_surv)
    if args.bag_loss == "cox_surv":
        return CoxSurvLoss()
    raise ValueError(f"Unsupported bag_loss: {args.bag_loss}")


def get_optimizer(model, args):
    params = filter(lambda param: param.requires_grad, model.parameters())
    if args.opt == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.reg)
    if args.opt == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    raise ValueError(f"Unsupported optimizer: {args.opt}")


class EarlyStoppingCIndex:
    def __init__(self, patience=20, stop_epoch=40):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, score, model, ckpt_name):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), ckpt_name)
            return

        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience and epoch > self.stop_epoch:
            self.early_stop = True


def unpack_model_output(output):
    if isinstance(output, dict):
        return output["hazards"], output["survival"], output.get("prediction")
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        prediction = output[2] if len(output) > 2 else None
        return output[0], output[1], prediction
    raise ValueError("Model output must be a dict or tuple containing hazards and survival.")


def forward_survival(model, batch, mode, device):
    if mode == "coattn":
        path = batch[0].to(device)
        omics = [tensor.to(device) for tensor in batch[1:7]]
        labels = batch[7].to(device)
        event_times = batch[8]
        censorships = batch[9].to(device)
        output = model(
            x_path=path,
            x_omic1=omics[0],
            x_omic2=omics[1],
            x_omic3=omics[2],
            x_omic4=omics[3],
            x_omic5=omics[4],
            x_omic6=omics[5],
        )
    else:
        path, omic, labels, event_times, censorships = batch
        path = path.to(device)
        omic = omic.to(device)
        labels = labels.to(device)
        censorships = censorships.to(device)
        output = model(x_path=path, x_omic=omic)

    hazards, survival, prediction = unpack_model_output(output)
    return hazards, survival, prediction, labels, event_times, censorships


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.0
    risks, censorships, event_times = [], [], []
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        hazards, survival, _, labels, batch_event_times, batch_censorships = forward_survival(
            model, batch, args.mode, device
        )
        loss = loss_fn(hazards, survival, labels, batch_censorships)
        loss_value = float(loss.item())
        total_loss += loss_value

        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        risks.extend(risk.tolist())
        censorships.extend(batch_censorships.detach().cpu().numpy().tolist())
        event_times.extend(batch_event_times.tolist())

        (loss / args.gc).backward()
        if (batch_idx + 1) % args.gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    mean_loss = total_loss / max(len(loader), 1)
    c_index = survival_c_index(event_times, censorships, risks)
    print(f"Epoch {epoch}: train_loss={mean_loss:.4f}, train_c_index={c_index:.4f}")
    if writer:
        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("train/c_index", c_index, epoch)


def validate(epoch, model, loader, loss_fn, args, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    risks, censorships, event_times = [], [], []
    results = {}
    slide_ids = loader.dataset.slide_data["slide_id"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            hazards, survival, _, labels, batch_event_times, batch_censorships = forward_survival(
                model, batch, args.mode, device
            )
            loss = loss_fn(hazards, survival, labels, batch_censorships)
            total_loss += float(loss.item())

            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            risks.extend(risk.tolist())
            censorships.extend(batch_censorships.detach().cpu().numpy().tolist())
            event_times.extend(batch_event_times.tolist())

            slide_id = slide_ids.iloc[batch_idx]
            results[slide_id] = {
                "slide_id": np.array(slide_id),
                "risk": float(risk[0]),
                "disc_label": int(labels.detach().cpu().numpy()[0]),
                "survival": float(batch_event_times[0]),
                "censorship": float(batch_censorships.detach().cpu().numpy()[0]),
            }

    mean_loss = total_loss / max(len(loader), 1)
    c_index = survival_c_index(event_times, censorships, risks)
    print(f"Epoch {epoch}: val_loss={mean_loss:.4f}, val_c_index={c_index:.4f}")
    if writer:
        writer.add_scalar("val/loss", mean_loss, epoch)
        writer.add_scalar("val/c_index", c_index, epoch)
    return results, c_index


def train_fold(datasets, fold_idx: int, args: Namespace):
    print(f"\nTraining fold {fold_idx}")
    os.makedirs(args.results_dir, exist_ok=True)
    writer = None
    if args.log_data:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(os.path.join(args.results_dir, str(fold_idx)), flush_secs=15)

    train_split, val_split = datasets
    save_splits(datasets, ["train", "val"], os.path.join(args.results_dir, f"splits_{fold_idx}.csv"))

    args.n_classes = train_split.num_classes
    args.omic_input_dim = train_split.genomic_features.shape[1]
    args.omic_sizes = getattr(train_split, "omic_sizes", None)

    model = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = get_loss(args)
    optimizer = get_optimizer(model, args)
    train_loader = get_split_loader(
        train_split,
        training=True,
        weighted=args.weighted_sample,
        mode=args.mode,
        batch_size=args.batch_size,
    )
    val_loader = get_split_loader(val_split, training=False, mode=args.mode, batch_size=args.batch_size)
    early_stopping = EarlyStoppingCIndex() if args.early_stopping else None

    latest_results, latest_c_index = {}, 0.0
    for epoch in range(args.max_epochs):
        train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, args, writer)
        latest_results, latest_c_index = validate(epoch, model, val_loader, loss_fn, args, writer)
        if early_stopping:
            ckpt_name = os.path.join(args.results_dir, f"s_{fold_idx}_checkpoint.pt")
            early_stopping(epoch, latest_c_index, model, ckpt_name)
            if early_stopping.early_stop:
                break

    if writer:
        writer.close()
    return latest_results, latest_c_index


def run_cross_validation(args: Namespace):
    seed_torch(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    dataset_csv = os.path.join(args.dataset_dir, f"{args.split_dir}_all_clean.csv.zip")
    split_root = os.path.join("splits", args.which_splits, args.split_dir)
    if not os.path.isfile(dataset_csv):
        raise FileNotFoundError(f"Missing dataset CSV: {dataset_csv}")
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    dataset = GenericMILSurvivalDataset(
        csv_path=dataset_csv,
        data_root_dir=args.data_root_dir,
        split_dir_name=args.split_dir,
        dataset_dir=args.dataset_dir,
        mode=args.mode,
        apply_sig=args.apply_sig,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_bins,
    )

    start = 0 if args.k_start < 0 else args.k_start
    end = args.k if args.k_end < 0 else args.k_end
    folds = np.arange(start, end)

    result_dir_name = f"{args.split_dir}_{args.model_type}_{args.bag_loss}_{args.which_splits}_s{args.seed}"
    args.results_dir = os.path.join(args.results_dir, args.which_splits, result_dir_name)
    os.makedirs(args.results_dir, exist_ok=True)

    latest_c_indices = []
    for fold_idx in folds:
        result_path = os.path.join(args.results_dir, f"split_latest_val_{fold_idx}_results.pkl")
        if os.path.isfile(result_path) and not args.overwrite:
            print(f"Skipping fold {fold_idx}; result already exists.")
            continue

        split_csv = os.path.join(split_root, f"splits_{fold_idx}.csv")
        train_split, val_split = dataset.return_splits(csv_path=split_csv)
        results, c_index = train_fold((train_split, val_split), int(fold_idx), args)
        save_pkl(result_path, results)
        latest_c_indices.append(c_index)

    summary = pd.DataFrame({"folds": folds[: len(latest_c_indices)], "val_cindex": latest_c_indices})
    if len(latest_c_indices) > 0:
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    {"folds": ["mean", "std"], "val_cindex": [summary["val_cindex"].mean(), summary["val_cindex"].std()]}
                ),
            ],
            ignore_index=True,
        )
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    return summary
