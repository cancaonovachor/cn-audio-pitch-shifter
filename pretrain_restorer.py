#!/usr/bin/env python3
"""
複数音源を使った事前学習スクリプト（戦略B Phase 1）。

使い方:
  python pretrain_restorer.py ../cn-youtube-downloader/downloads/ \\
      -o pretrained.pt --epochs 200

処理の流れ:
  1. 指定ディレクトリの全音源からラウンドトリップ学習データを生成
  2. サンプルを .restorer_cache/ にキャッシュ（再実行時はスキップ）
  3. 全サンプルを使って U-Net を学習
"""

import argparse
import glob
import hashlib
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import stft as _stft
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model_restorer import HOP_LENGTH, N_FFT, PitchRestorer

SR = 44100
CHUNK_N = int(2.0 * SR)
STRIDE_N = CHUNK_N // 2
EPS = 1e-8
SHIFT_AMOUNTS = [4, 5, 6]
DEFAULT_CACHE = os.path.join(SCRIPT_DIR, ".restorer_cache")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


# ---------------------------------------------------------------------------
# 音源ロード
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp3", ".m4a", ".ogg"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "44100",
                 "-f", "wav", "-acodec", "pcm_f32le", tmp],
                check=True, capture_output=True,
            )
            y, _ = sf.read(tmp, dtype="float32")
        finally:
            os.unlink(tmp)
    else:
        y, sr = sf.read(path, dtype="float32")
        if sr != SR:
            import soxr
            y = soxr.resample(y, sr, SR, quality="VHQ")
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# STFT ユーティリティ
# ---------------------------------------------------------------------------

def _rb_shift(y: np.ndarray, n: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        in_p = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_p = f.name
    try:
        sf.write(in_p, y, SR, subtype="FLOAT")
        subprocess.run(
            ["rubberband", "--fine", "--pitch", str(n), in_p, out_p],
            check=True, capture_output=True,
        )
        out, _ = sf.read(out_p, dtype="float32")
    finally:
        os.unlink(in_p)
        os.unlink(out_p)
    return out[: len(y)].astype(np.float32)


def _to_logmag(y: np.ndarray) -> np.ndarray:
    _, _, Z = _stft(y, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    return np.log(np.abs(Z) + EPS).astype(np.float32)


def _pad8(spec: np.ndarray) -> np.ndarray:
    T = spec.shape[1]
    T8 = ((T + 7) // 8) * 8
    return np.pad(spec, ((0, 0), (0, T8 - T))) if T8 > T else spec


# ---------------------------------------------------------------------------
# ディスクキャッシュ生成
# ---------------------------------------------------------------------------

def _file_hash(path: str) -> str:
    h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    return h


def build_cache(audio_path: str, cache_dir: str) -> list[str]:
    """
    1音源分のキャッシュを生成して .npz ファイルのパスリストを返す。
    キャッシュ済みの場合はスキップ。
    """
    fhash = _file_hash(audio_path)
    song_dir = os.path.join(cache_dir, fhash)
    done_flag = os.path.join(song_dir, "done.txt")

    if os.path.exists(done_flag):
        paths = sorted(glob.glob(os.path.join(song_dir, "*.npz")))
        print(f"  [{os.path.basename(audio_path)}] キャッシュ済み ({len(paths)} サンプル)")
        return paths

    os.makedirs(song_dir, exist_ok=True)
    print(f"  [{os.path.basename(audio_path)}] データ生成中...")

    try:
        y_clean = _load_audio(audio_path)
    except Exception as e:
        print(f"    ロード失敗: {e}", file=sys.stderr)
        return []

    paths = []
    idx = 0
    for n in SHIFT_AMOUNTS:
        for sign in (+1, -1):
            print(f"    ラウンドトリップ: {sign*n:+d} → {-sign*n:+d} 半音 ...", flush=True)
            y_deg = _rb_shift(_rb_shift(y_clean, sign * n), -sign * n)
            wav_len = min(len(y_clean), len(y_deg))

            for start in range(0, wav_len - CHUNK_N, STRIDE_N):
                c_deg   = y_deg  [start: start + CHUNK_N]
                c_clean = y_clean[start: start + CHUNK_N]
                X = _pad8(_to_logmag(c_deg)).astype(np.float16)
                Y = _pad8(_to_logmag(c_clean)).astype(np.float16)
                npz_path = os.path.join(song_dir, f"{idx:05d}.npz")
                np.savez_compressed(npz_path, X=X, Y=Y)
                paths.append(npz_path)
                idx += 1

    with open(done_flag, "w") as f:
        f.write(str(len(paths)))
    print(f"    → {len(paths)} サンプルをキャッシュ")
    return sorted(paths)


# ---------------------------------------------------------------------------
# Dataset（ディスク遅延ロード + 並列I/O）
# ---------------------------------------------------------------------------

class CachedPairDataset(Dataset):
    def __init__(self, paths: list[str], augment: bool):
        self.paths = paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        d = np.load(self.paths[idx])
        X = d["X"].astype(np.float32)
        Y = d["Y"].astype(np.float32)
        if self.augment:
            g = np.random.uniform(np.log(0.45), 0.0)
            X = X + g
            Y = Y + g
        return torch.from_numpy(X[np.newaxis]), torch.from_numpy(Y[np.newaxis])


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(args: argparse.Namespace) -> None:
    # 音源ファイル一覧
    input_path = args.input
    if os.path.isdir(input_path):
        audio_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS
        ])
    else:
        audio_files = [input_path]

    print(f"音源ファイル数: {len(audio_files)}")

    # キャッシュ生成
    os.makedirs(args.cache_dir, exist_ok=True)
    all_paths: list[str] = []
    for i, af in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {os.path.basename(af)}")
        all_paths.extend(build_cache(af, args.cache_dir))

    print(f"\n合計 {len(all_paths)} サンプル")

    # Train / Val 分割（8:2）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_paths))
    n_val = max(1, len(all_paths) // 5)
    tr_paths = [all_paths[i] for i in idx[n_val:]]
    va_paths = [all_paths[i] for i in idx[:n_val]]
    print(f"Train: {len(tr_paths)}  Val: {len(va_paths)}")

    tr_dl = DataLoader(CachedPairDataset(tr_paths, augment=True),
                       batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=4, persistent_workers=True)
    va_dl = DataLoader(CachedPairDataset(va_paths, augment=False),
                       batch_size=args.batch, shuffle=False,
                       num_workers=4, persistent_workers=True)

    print("デバイス初期化中...", flush=True)
    device = _get_device()
    print(f"デバイス: {device}  モデル初期化中...", flush=True)
    model = PitchRestorer().to(device)
    print(f"モデル初期化完了", flush=True)

    if args.finetune:
        ck = torch.load(args.finetune, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"Fine-Tune元: {args.finetune}  (val_loss={ck['val_loss']:.4f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}", flush=True)

    lr = args.lr if args.lr is not None else (1e-5 if args.finetune else 6e-4)
    print(f"学習率: {lr:.1e}  最適化器・スケジューラ初期化中...", flush=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5, min_lr=1e-6
    )
    crit  = nn.L1Loss()
    print(f"学習準備完了。エポック開始...", flush=True)

    best_val   = float("inf")   # チェックポイント用（生のval_loss）
    best_ema   = float("inf")   # 早期終了用（EMA平滑後）
    ema_val    = None            # EMA値（初回はval_lossで初期化）
    no_improve = 0
    out_path   = os.path.join(SCRIPT_DIR, args.output)
    epoch_times: list[float] = []
    t_start = time.time()

    def _eta(remaining: int) -> str:
        if not epoch_times:
            return "算出中"
        avg = sum(epoch_times[-10:]) / len(epoch_times[-10:])
        secs = int(avg * remaining)
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}分{s:02d}秒"

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()

        model.train()
        tr_loss = 0.0
        for X, Y in tr_dl:
            X, Y = X.to(device), Y.to(device)
            loss = crit(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, Y in va_dl:
                va_loss += crit(model(X.to(device)), Y.to(device)).item()
        va_loss /= len(va_dl)

        # EMA平滑化（α=0.2 ≒ 5epoch窓）
        ema_val = va_loss if ema_val is None else 0.8 * ema_val + 0.2 * va_loss

        # LRスケジューラ・早期終了はEMAで判断（スパイクに騙されない）
        sched.step(ema_val)
        epoch_times.append(time.time() - t_ep)
        lr_now = opt.param_groups[0]["lr"]

        # チェックポイントは生のval_lossが最小のモデルを保存
        ckpt_marker = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"epoch": ep, "state": model.state_dict(),
                        "val_loss": va_loss}, out_path)
            ckpt_marker = " [保存]"

        # 早期終了判定はEMAベース
        if ema_val < best_ema:
            best_ema   = ema_val
            no_improve = 0
            stop_marker = f"  ★EMA更新{ckpt_marker}"
        else:
            no_improve += 1
            eta = _eta(args.patience - no_improve)
            stop_marker = (f"  [改善なし {no_improve:2d}/{args.patience}]"
                           f"  残り {eta}{ckpt_marker}")
            if no_improve >= args.patience:
                print(f"  Epoch {ep:4d}/{args.epochs}"
                      f"  train={tr_loss:.4f}  val={va_loss:.4f}"
                      f"  ema={ema_val:.4f}  lr={lr_now:.1e}{stop_marker}")
                print(f"  早期終了 (EMAベース patience={args.patience}, epoch={ep})")
                break

        print(f"  Epoch {ep:4d}/{args.epochs}"
              f"  train={tr_loss:.4f}  val={va_loss:.4f}"
              f"  ema={ema_val:.4f}  lr={lr_now:.1e}{stop_marker}", flush=True)

    elapsed = int(time.time() - t_start)
    h, r = divmod(elapsed, 3600)
    m, s = divmod(r, 60)
    print(f"完了。ベストモデル: {out_path}  (val_loss={best_val:.4f})"
          f"  経過時間: {h}:{m:02d}:{s:02d}")


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="複数音源による事前学習")
    ap.add_argument("input",
                    help="音源ディレクトリ（MP3/WAV/FLAC）またはファイル1つ")
    ap.add_argument("-o", "--output", default="pretrained.pt",
                    help="保存先モデルファイル (default: pretrained.pt)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE,
                    help=f"サンプルキャッシュ先 (default: {DEFAULT_CACHE})")
    ap.add_argument("--epochs",   type=int,   default=200)
    ap.add_argument("--batch",    type=int,   default=64,
                    help="バッチサイズ (default: 64、LRはsqrt比例で6e-4)")
    ap.add_argument("--lr",       type=float, default=None,
                    help="学習率 (default: 6e-4、--finetune 時は 1e-5)")
    ap.add_argument("--patience", type=int,   default=25,
                    help="EMAベース早期終了 patience (default: 25)")
    ap.add_argument("--finetune", metavar="MODEL_PT",
                    help="既存モデルから Fine-Tune（重みを引き継ぎ）")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
