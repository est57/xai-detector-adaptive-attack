from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datasets import SPECS


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    raw_min: np.ndarray
    raw_max: np.ndarray
    log_mask: np.ndarray
    spec: object


def load_raw(spec, path):
    """Membaca file dataset dan mengubah label menjadi biner (0 = normal, 1 = serangan)."""
    if spec.columns is None:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=spec.columns)
    df[spec.label_col] = (df[spec.label_col] != spec.normal_label).astype(np.int64)
    return df.drop(columns=spec.drop)


def preprocess(train_df, test_df, spec):
    """One-hot fitur kategorikal, log1p fitur bervolume besar, lalu min-max scaling ke [0, 1]."""
    y_train = train_df.pop(spec.label_col).to_numpy().copy()
    y_test = test_df.pop(spec.label_col).to_numpy().copy()

    full = pd.get_dummies(pd.concat([train_df, test_df], keys=["train", "test"]),
                          columns=spec.categorical, dtype=np.float32)
    train_x, test_x = full.xs("train").copy(), full.xs("test").copy()

    for col in spec.log_features:
        train_x[col] = np.log1p(train_x[col])
        test_x[col] = np.log1p(test_x[col])

    raw_min = train_x.min().to_numpy(dtype=np.float32)
    raw_max = train_x.max().to_numpy(dtype=np.float32)
    span = np.where(raw_max - raw_min > 0, raw_max - raw_min, 1.0)

    x_train = ((train_x.to_numpy(dtype=np.float32) - raw_min) / span).clip(0, 1)
    x_test = ((test_x.to_numpy(dtype=np.float32) - raw_min) / span).clip(0, 1)

    names = list(train_x.columns)
    log_mask = np.array([n in spec.log_features for n in names])
    return Dataset(x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test,
                   names, raw_min, raw_max, log_mask, spec)


def load_dataset(name):
    """Fungsi utama untuk memuat dan memproses dataset berdasarkan nama."""
    spec = SPECS[name]
    return preprocess(load_raw(spec, spec.train_file), load_raw(spec, spec.test_file), spec)
