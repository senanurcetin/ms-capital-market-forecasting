"""A learned sequence representation over the 600 s market window.

WHY THIS EXISTS. The README's sequence-order experiment added eighteen hand-built path
statistics to the 292 features and gained +0.0006 with a CI spanning zero. That falsifies
those statistics, not the hypothesis: the nine novel ones score 0.0057 standing alone, which
is where a constant predictor sits. Telling "the path holds no signal" from "eighteen numbers
were the wrong way to look at it" needs a representation learned from the ~176 snapshots
themselves, in the spirit of DeepLOB (Zhang et al., 2019).

WHAT IT TESTS. Not "is a CNN any good" but "does it know something the tabular model does
not". Both models see the same samples, and the question is whether blending the CNN into the
LightGBM prediction improves held-out months. The blend weights are fitted on validation
months only; the test months are never touched until the end.

The sample is one window of snapshots ordered oldest -> newest (`seconds_before_predict`
descending), so the tensor is (n_samples, WINDOW, N_CHANNELS) with the newest snapshot last.
Everything that touches NumPy only lives at the top of this file and is unit-tested; the
network and the training loop import PyTorch lazily, because PyTorch is not a dependency of
the serving image or of CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 176  # mean rows per sample in the market table (README: 176.3)

CHANNELS = (
    "mid_ret_bps", "spread_bps", "log_bid_vol1", "log_ask_vol1", "log_bid_vol2",
    "log_ask_vol2", "imb_l1", "imb_l12", "micro_off_bps", "ask_gap_bps", "bid_gap_bps",
    "txn_off_bps", "log_txn_vol", "log_txn_count", "age", "valid",
)
N_CHANNELS = len(CHANNELS)
RAW_COLUMNS = (
    "sample_id", "seconds_before_predict", "transaction_avgprice", "transaction_volume",
    "transaction_count", "ask_price_1", "ask_volume_1", "bid_price_1", "bid_volume_1",
    "ask_price_2", "ask_volume_2", "bid_price_2", "bid_volume_2",
)
_EPS = 1e-9


def _mid(df: pd.DataFrame) -> np.ndarray:
    """Mid price per snapshot, float64. See `snapshot_channels` for the empty-level rule."""
    ask1, bid1 = df["ask_price_1"].to_numpy(np.float64), df["bid_price_1"].to_numpy(np.float64)
    sid = df["sample_id"].to_numpy()
    mid = pd.Series(np.where((ask1 > 0) & (bid1 > 0), (ask1 + bid1) / 2, np.nan))
    return mid.groupby(sid).ffill().groupby(sid).bfill().fillna(1.0).to_numpy()


def snapshot_channels(df: pd.DataFrame) -> np.ndarray:
    """One row of raw order-book state -> N_CHANNELS numbers, for every row of `df`.

    `df` must be sorted by (sample_id, seconds_before_predict descending). A price of 0 is the
    table's "empty level" sentinel (README: always paired with volume 0), so a mid built from
    it would be garbage; it is treated as missing and carried forward from the previous
    snapshot of the same sample. A sample whose book is empty throughout falls back to 1.0,
    the normalisation point of the price columns.

    `valid` and `mid_ret_bps` are filled by `build_tensor`, because they depend on where each
    sample's window starts and ends.
    """
    f = lambda c: df[c].to_numpy(dtype=np.float64)  # noqa: E731
    ask1, bid1, ask2, bid2 = f("ask_price_1"), f("bid_price_1"), f("ask_price_2"), f("bid_price_2")
    av1, bv1, av2, bv2 = f("ask_volume_1"), f("bid_volume_1"), f("ask_volume_2"), f("bid_volume_2")
    ok = (ask1 > 0) & (bid1 > 0)
    mid = _mid(df)

    spread = np.where(ok, (ask1 - bid1) / mid * 1e4, 0.0)
    imb1 = (bv1 - av1) / (bv1 + av1 + _EPS)
    imb12 = ((bv1 + bv2) - (av1 + av2)) / (bv1 + bv2 + av1 + av2 + _EPS)
    micro = np.where(ok & (bv1 + av1 > 0),
                     (ask1 * bv1 + bid1 * av1) / (bv1 + av1 + _EPS) / mid - 1.0, 0.0) * 1e4
    ask_gap = np.where((ask1 > 0) & (ask2 > 0), (ask2 - ask1) / mid * 1e4, 0.0)
    bid_gap = np.where((bid1 > 0) & (bid2 > 0), (bid1 - bid2) / mid * 1e4, 0.0)
    tc = f("transaction_count")
    txn_off = np.where(tc > 0, (f("transaction_avgprice") / mid - 1.0) * 1e4, 0.0)

    out = np.empty((len(df), N_CHANNELS), dtype=np.float32)
    out[:, 0] = 0.0  # filled by build_tensor: the return against the newest mid
    out[:, 1] = spread
    out[:, 2], out[:, 3] = np.log1p(np.maximum(bv1, 0)), np.log1p(np.maximum(av1, 0))
    out[:, 4], out[:, 5] = np.log1p(np.maximum(bv2, 0)), np.log1p(np.maximum(av2, 0))
    out[:, 6], out[:, 7], out[:, 8] = imb1, imb12, micro
    out[:, 9], out[:, 10], out[:, 11] = ask_gap, bid_gap, txn_off
    out[:, 12] = np.log1p(np.maximum(f("transaction_volume"), 0))
    out[:, 13] = np.log1p(np.maximum(tc, 0))
    out[:, 14] = f("seconds_before_predict") / 600.0
    out[:, 15] = 1.0
    return out


def build_tensor(df: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Raw snapshots -> (sample_ids, tensor of shape (n, window, N_CHANNELS)).

    Each sample keeps its NEWEST `window` snapshots, newest last. A sample with fewer is
    left-padded by repeating its oldest snapshot, with `valid` = 0 on the padding so the
    network can tell a real flat stretch from a filled one. `mid_ret_bps` is the mid's return
    against the sample's newest mid: the level of an anonymised price carries no meaning, the
    path to the prediction instant does.
    """
    if not df["sample_id"].is_monotonic_increasing:
        raise ValueError("df must be sorted by sample_id (then seconds_before_predict desc)")
    sid = df["sample_id"].to_numpy()
    ch = snapshot_channels(df)
    mid = _mid(df)

    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    n = len(starts)
    counts = ends - starts
    group = np.repeat(np.arange(n), counts)
    from_end = (ends[group] - 1) - np.arange(len(sid))  # 0 = newest snapshot
    keep = from_end < window

    X = np.zeros((n, window, N_CHANNELS), dtype=np.float32)
    X[group[keep], window - 1 - from_end[keep]] = ch[keep]
    M = np.ones((n, window), dtype=np.float64)  # float64: a price level in float32 loses bps
    M[group[keep], window - 1 - from_end[keep]] = mid[keep]

    first = window - np.minimum(counts, window)  # index of the oldest real snapshot
    idx = np.maximum(np.arange(window)[None, :], first[:, None])
    X = np.take_along_axis(X, idx[:, :, None], axis=1)
    M = np.take_along_axis(M, idx, axis=1)
    X[:, :, 15] = (np.arange(window)[None, :] >= first[:, None]).astype(np.float32)

    X[:, :, 0] = ((M / M[:, -1:] - 1.0) * 1e4).astype(np.float32)
    return sid[starts], X


def build_tensor_chunked(table, window: int = WINDOW, *, rows_per_chunk: int = 2_000_000,
                         out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`build_tensor` over an Arrow table too large to convert whole.

    The intermediate arrays are float64 and about twenty of them are alive at once, so 32M
    rows cost ~13 GB and the build was killed by the memory limit of a 16 GB machine. Rows are
    cut into chunks at sample boundaries and each chunk is built on its own. Nothing in the
    builder looks across samples, so the result is identical to the one-piece build
    (tests/test_sequence_cnn.py compares them). `out` can be a memory-mapped array to keep the
    tensor itself off the heap.
    """
    sid = table.column("sample_id").to_numpy()
    n_samples = int(np.count_nonzero(np.r_[True, sid[1:] != sid[:-1]]))
    X = out if out is not None else np.empty((n_samples, window, N_CHANNELS), dtype=np.float32)
    ids = np.empty(n_samples, dtype=sid.dtype)
    row, filled = 0, 0
    while row < len(sid):
        end = min(row + rows_per_chunk, len(sid))
        if end < len(sid):  # move the cut forward to the next sample boundary
            end = int(np.searchsorted(sid, sid[end - 1], side="right"))
        sub_ids, sub = build_tensor(table.slice(row, end - row).to_pandas(), window)
        X[filled:filled + len(sub_ids)], ids[filled:filled + len(sub_ids)] = sub, sub_ids
        row, filled = end, filled + len(sub_ids)
    return ids, X


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std from TRAINING samples only. `valid` is left unscaled."""
    flat = X.reshape(-1, X.shape[-1])
    sub = flat[:: max(1, len(flat) // 2_000_000)]
    mu, sd = sub.mean(0), sub.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    mu[-1], sd[-1] = 0.0, 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """Standardise and clip, in place on a copy of the channel axis (heavy tails otherwise)."""
    return np.clip((X - mu) / sd, -clip, clip).astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------------
# Everything below needs PyTorch.
# --------------------------------------------------------------------------------------------
def make_network(n_channels: int = N_CHANNELS, width: int = 48):
    """A small dilated 1-D CNN: three residual blocks, then pooled statistics -> one number."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c: int, dilation: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation)
            self.n1, self.n2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
            self.act = nn.GELU()

        def forward(self, x):
            h = self.conv1(self.act(self.n1(x)))
            h = self.conv2(self.act(self.n2(h)))
            return x + h

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv1d(n_channels, width, 5, padding=2)
            self.blocks = nn.Sequential(Block(width, 1), Block(width, 2), Block(width, 4))
            self.norm = nn.BatchNorm1d(width)
            self.head = nn.Sequential(nn.Linear(3 * width, 64), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, x):  # x: (batch, window, channels)
            h = self.blocks(self.stem(x.transpose(1, 2)))
            h = torch.nn.functional.gelu(self.norm(h))
            pooled = torch.cat([h.mean(-1), h.amax(-1), h[:, :, -8:].mean(-1)], dim=1)
            return self.head(pooled).squeeze(-1)

    return Net()


def predict(net, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[i:i + batch])).numpy())
    return np.concatenate(out).astype(np.float64)


def train(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, *,
          epochs: int = 12, patience: int = 3, lr: float = 2e-3, batch: int = 512,
          seed: int = 0, log=print):
    """Fit on the training months, keep the epoch with the best validation cosine.

    The target is divided by its training std and clipped at +-5: it is heavy-tailed, and a
    squared loss otherwise spends the whole fit on a few outliers. Cosine is only the
    *selection* criterion here, because the README shows that training on a cosine-shaped loss
    made the gradient-boosted models worse.
    """
    import copy

    import torch

    from src.evaluation.metrics import cosine_similarity

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(y_tr.std())
    t_tr = torch.from_numpy(np.clip(y_tr / scale, -5, 5).astype(np.float32))
    net = make_network(X_tr.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(X_tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    best, best_state, bad, history = -np.inf, None, 0, []
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(X_tr))
        total = 0.0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(torch.from_numpy(X_tr[b])), t_tr[b])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
        val = cosine_similarity(y_va, predict(net, X_va))
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_cosine": val})
        log(f"epoch {epoch + 1:2d}  train loss {total / len(order):.4f}  val cosine {val:+.5f}")
        if val > best:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, history
