from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

import config

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]


@dataclass
class DatasetSpec:
    """Deskripsi dataset: file, kolom, preprocessing, dan batasan domain."""
    name: str
    train_file: object
    test_file: object
    columns: Optional[list]
    label_col: str
    normal_label: object
    drop: list
    categorical: list
    log_features: list
    increase_only: list
    free: list
    integer: list
    derived: list = field(default_factory=list)
    derive: Optional[Callable] = None


def safe_div(a, b):
    return torch.where(b > 0, a / b.clamp(min=1e-9), torch.zeros_like(a))


def unsw_derive(raw, raw_orig, idx):
    """Perbaikan dan penghitungan ulang fitur turunan UNSW-NB15 agar konsisten dengan fitur primer."""
    r = raw.clone()
    spkts, sbytes, dur, dpkts = idx["spkts"], idx["sbytes"], idx["dur"], idx["dpkts"]

    # Setiap paket tambahan minimal 40 byte (header TCP/IP), maksimal 1500 byte per paket (MTU)
    added_pkts = r[:, spkts] - raw_orig[:, spkts]
    low = raw_orig[:, sbytes] + 40 * added_pkts
    high = torch.maximum(1500 * r[:, spkts], raw_orig[:, sbytes])
    r[:, sbytes] = torch.round(torch.minimum(torch.maximum(r[:, sbytes], low), high))

    formulas = {
        "smean": lambda x: x[:, sbytes] / x[:, spkts].clamp(min=1),
        "sload": lambda x: safe_div((x[:, sbytes] - x[:, sbytes] / x[:, spkts].clamp(min=1)) * 8, x[:, dur]),
        "rate": lambda x: safe_div(x[:, spkts] + x[:, dpkts] - 1, x[:, dur]),
        "sinpkt": lambda x: x[:, dur] * 1000 / (x[:, spkts] - 1).clamp(min=1),
    }
    for name, g in formulas.items():
        r[:, idx[name]] = (raw_orig[:, idx[name]] + g(r) - g(raw_orig)).clamp(min=0)
    r[:, idx["smean"]] = torch.round(r[:, idx["smean"]])
    return r


NSL_KDD = DatasetSpec(
    name="nsl_kdd",
    train_file=config.DATA_DIR / "KDDTrain+.txt",
    test_file=config.DATA_DIR / "KDDTest+.txt",
    columns=NSL_KDD_COLUMNS,
    label_col="label",
    normal_label="normal",
    drop=["difficulty"],
    categorical=["protocol_type", "service", "flag"],
    log_features=["duration", "src_bytes", "dst_bytes"],
    increase_only=["duration", "src_bytes", "dst_bytes"],
    free=[
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ],
    integer=["duration", "src_bytes", "dst_bytes", "count", "srv_count",
             "dst_host_count", "dst_host_srv_count"],
)

UNSW_NB15 = DatasetSpec(
    name="unsw_nb15",
    train_file=config.DATA_DIR / "unsw_nb15" / "UNSW_NB15_training-set.csv",
    test_file=config.DATA_DIR / "unsw_nb15" / "UNSW_NB15_testing-set.csv",
    columns=None,
    label_col="label",
    normal_label=0,
    drop=["id", "attack_cat"],
    categorical=["proto", "service", "state"],
    log_features=["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
                  "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpb",
                  "smean", "dmean", "response_body_len"],
    increase_only=["dur", "spkts", "sbytes", "sjit",
                   "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                   "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"],
    free=[],
    integer=["spkts", "sbytes", "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm",
             "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"],
    derived=["smean", "sload", "rate", "sinpkt"],
    derive=unsw_derive,
)

SPECS = {s.name: s for s in (NSL_KDD, UNSW_NB15)}
