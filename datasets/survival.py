from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


METADATA_COLUMNS = {
    "case_id",
    "slide_id",
    "site",
    "is_female",
    "oncotree_code",
    "age",
    "survival_months",
    "censorship",
    "train",
    "disc_label",
    "label",
}


def save_splits(split_datasets: Iterable[Dataset], column_keys: list[str], filename: str) -> None:
    splits = [split.slide_data["slide_id"] for split in split_datasets]
    df = pd.concat(splits, ignore_index=True, axis=1)
    df.columns = column_keys
    df.to_csv(filename, index=False)


def _strip_slide_suffix(slide_id: str) -> str:
    return slide_id[:-4] if slide_id.endswith(".svs") else slide_id


def _study_from_split_name(split_dir: str) -> str:
    parts = split_dir.split("_")
    if len(parts) < 2:
        raise ValueError(f"Expected split_dir like 'tcga_blca', got {split_dir!r}.")
    return "_".join(parts[:2])


def _load_slide_features(data_dir: str, slide_ids: np.ndarray) -> torch.Tensor:
    features = []
    for slide_id in slide_ids:
        feature_path = os.path.join(data_dir, "pt_files", f"{_strip_slide_suffix(str(slide_id))}.pt")
        if not os.path.isfile(feature_path):
            raise FileNotFoundError(
                f"Missing WSI feature file: {feature_path}. "
                "Check --data_root_dir and the expected feature layout."
            )
        features.append(torch.load(feature_path, map_location="cpu"))
    return torch.cat(features, dim=0)


@dataclass
class SurvivalBins:
    edges: np.ndarray
    num_bins: int


class GenericMILSurvivalDataset(Dataset):
    """Patient-level survival dataset backed by pre-extracted WSI features."""

    def __init__(
        self,
        csv_path: str,
        data_root_dir: str,
        split_dir_name: str,
        dataset_dir: str = "dataset_csv",
        mode: str = "coattn",
        apply_sig: bool = True,
        shuffle: bool = False,
        seed: int = 1,
        print_info: bool = True,
        n_bins: int = 4,
        label_col: str = "survival_months",
        censorship_col: str = "censorship",
    ) -> None:
        self.csv_path = csv_path
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.apply_sig = apply_sig
        self.seed = seed
        self.label_col = label_col
        self.censorship_col = censorship_col
        self.study = _study_from_split_name(split_dir_name)
        self.data_dir = os.path.join(data_root_dir, self.study, "UNI")

        slide_data = pd.read_csv(csv_path, low_memory=False)
        if "case_id" not in slide_data.columns:
            slide_data.index = slide_data.index.astype(str).str[:12]
            slide_data["case_id"] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        if label_col not in slide_data.columns:
            raise ValueError(f"Missing label column {label_col!r} in {csv_path}.")
        if censorship_col not in slide_data.columns:
            raise ValueError(f"Missing censorship column {censorship_col!r} in {csv_path}.")

        if "oncotree_code" in slide_data.columns and (slide_data["oncotree_code"] == "IDC").any():
            slide_data = slide_data[slide_data["oncotree_code"] == "IDC"].copy()

        if shuffle:
            slide_data = slide_data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.slide_data, self.patient_dict, self.bins = self._prepare_patient_table(
            slide_data, n_bins, label_col, censorship_col
        )
        self.num_classes = self.bins.num_bins
        self.patient_data = {
            "case_id": self.slide_data["case_id"].values,
            "label": self.slide_data["label"].values,
        }
        self.slide_cls_ids = [
            np.where(self.slide_data["label"].values == class_id)[0] for class_id in range(self.slide_data["label"].nunique())
        ]

        self.signatures = self._load_signatures() if apply_sig else None
        self.genomic_columns = [col for col in self.slide_data.columns if col not in METADATA_COLUMNS]

        if print_info:
            self.summarize()

    @staticmethod
    def _prepare_patient_table(
        slide_data: pd.DataFrame,
        n_bins: int,
        label_col: str,
        censorship_col: str,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], SurvivalBins]:
        patients_df = slide_data.drop_duplicates(["case_id"]).copy()
        uncensored_df = patients_df[patients_df[censorship_col] < 1]
        if uncensored_df.empty:
            raise ValueError("At least one uncensored patient is required to build survival bins.")

        _, q_bins = pd.qcut(
            uncensored_df[label_col],
            q=n_bins,
            retbins=True,
            labels=False,
            duplicates="drop",
        )
        q_bins[0] = slide_data[label_col].min() - 1e-6
        q_bins[-1] = slide_data[label_col].max() + 1e-6
        disc_labels = pd.cut(
            patients_df[label_col],
            bins=q_bins,
            labels=False,
            right=False,
            include_lowest=True,
        )
        patients_df.insert(2, "disc_label", disc_labels.astype(int))

        slide_lookup = slide_data.set_index("case_id")
        patient_dict = {}
        for case_id in patients_df["case_id"]:
            slide_ids = slide_lookup.loc[case_id, "slide_id"]
            if isinstance(slide_ids, str):
                slide_ids = np.array([slide_ids])
            else:
                slide_ids = slide_ids.values
            patient_dict[case_id] = slide_ids

        patients_df = patients_df.assign(slide_id=patients_df["case_id"])
        patients_df["label"] = patients_df["disc_label"].astype(int)
        patients_df = patients_df.reset_index(drop=True)
        return patients_df, patient_dict, SurvivalBins(edges=q_bins, num_bins=len(q_bins) - 1)

    def _load_signatures(self) -> pd.DataFrame:
        signature_path = os.path.join(self.dataset_dir, "signatures.csv")
        if not os.path.isfile(signature_path):
            raise FileNotFoundError(f"Missing signature file: {signature_path}.")
        return pd.read_csv(signature_path)

    def summarize(self) -> None:
        print(f"Dataset: {self.study}")
        print(f"Patients: {len(self.slide_data)}")
        print(f"Survival bins: {self.bins.edges.tolist()}")
        print("Discrete label counts:")
        print(self.slide_data["label"].value_counts(sort=False))

    def __len__(self) -> int:
        return len(self.slide_data)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Use return_splits() to create train/validation split datasets.")

    def get_split_from_df(self, all_splits: pd.DataFrame, split_key: str) -> "GenericSplit":
        split_ids = all_splits[split_key].dropna().reset_index(drop=True)
        mask = self.slide_data["slide_id"].isin(split_ids.tolist())
        split_data = self.slide_data[mask].reset_index(drop=True)
        return GenericSplit(
            slide_data=split_data,
            mode=self.mode,
            signatures=self.signatures,
            data_dir=self.data_dir,
            label_col=self.label_col,
            censorship_col=self.censorship_col,
            patient_dict=self.patient_dict,
            genomic_columns=self.genomic_columns,
            num_classes=self.num_classes,
        )

    def return_splits(self, csv_path: str) -> tuple["GenericSplit", "GenericSplit"]:
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits, "train")
        val_split = self.get_split_from_df(all_splits, "val")

        scalers = train_split.get_scaler()
        train_split.apply_scaler(scalers)
        val_split.apply_scaler(scalers)
        return train_split, val_split


class GenericSplit(Dataset):
    """Split dataset that materializes tensors for one patient at a time."""

    def __init__(
        self,
        slide_data: pd.DataFrame,
        mode: str,
        signatures: pd.DataFrame | None,
        data_dir: str,
        label_col: str,
        censorship_col: str,
        patient_dict: dict[str, np.ndarray],
        genomic_columns: list[str],
        num_classes: int,
    ) -> None:
        self.slide_data = slide_data
        self.mode = mode
        self.signatures = signatures
        self.data_dir = data_dir
        self.label_col = label_col
        self.censorship_col = censorship_col
        self.patient_dict = patient_dict
        self.genomic_columns = genomic_columns
        self.num_classes = num_classes
        self.genomic_features = self.slide_data[self.genomic_columns].copy()
        self.slide_cls_ids = [
            np.where(self.slide_data["label"].values == class_id)[0] for class_id in range(self.num_classes)
        ]
        self.omic_names = self._build_signature_groups() if signatures is not None else []
        self.omic_sizes = [len(names) for names in self.omic_names]

    def _build_signature_groups(self) -> list[list[str]]:
        groups = []
        genomic_columns = set(self.genomic_features.columns)
        for col in self.signatures.columns:
            genes = self.signatures[col].dropna().unique()
            candidates = np.concatenate([genes + suffix for suffix in ["_mut", "_cnv", "_rnaseq"]])
            groups.append(sorted(set(candidates) & genomic_columns))
        return groups

    def __len__(self) -> int:
        return len(self.slide_data)

    def getlabel(self, idx: int) -> int:
        return int(self.slide_data.loc[idx, "label"])

    def get_scaler(self) -> tuple[StandardScaler]:
        return (StandardScaler().fit(self.genomic_features),)

    def apply_scaler(self, scalers: tuple[StandardScaler]) -> None:
        transformed = pd.DataFrame(
            scalers[0].transform(self.genomic_features),
            columns=self.genomic_features.columns,
        )
        self.genomic_features = transformed

    def __getitem__(self, idx: int):
        row = self.slide_data.loc[idx]
        case_id = row["case_id"]
        label = int(row["disc_label"])
        event_time = float(row[self.label_col])
        censorship = float(row[self.censorship_col])
        slide_ids = self.patient_dict[case_id]

        if self.mode == "omic":
            genomic_features = torch.tensor(self.genomic_features.iloc[idx].values, dtype=torch.float32)
            return torch.zeros((1, 1), dtype=torch.float32), genomic_features, label, event_time, censorship

        path_features = _load_slide_features(self.data_dir, slide_ids)

        if self.mode == "path":
            return path_features, torch.zeros((1, 1), dtype=torch.float32), label, event_time, censorship

        if self.mode == "pathomic":
            genomic_features = torch.tensor(self.genomic_features.iloc[idx].values, dtype=torch.float32)
            return path_features, genomic_features, label, event_time, censorship

        if self.mode == "coattn":
            if len(self.omic_names) == 0:
                raise ValueError("Mode 'coattn' requires genomic signature groups.")
            omics = [
                torch.tensor(self.genomic_features[names].iloc[idx].values, dtype=torch.float32)
                for names in self.omic_names
            ]
            return (path_features, *omics, label, event_time, censorship)

        raise NotImplementedError(f"Unsupported mode: {self.mode}")
