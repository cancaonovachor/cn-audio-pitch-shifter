#!/usr/bin/env python3
"""
ピッチシフト劣化復元モデルの学習スクリプト。

ラウンドトリップ戦略:
  原音 → rubberband −n半音 → rubberband +n半音 → 劣化した原音  (両方向 × 複数量)
  学習ペア: (劣化した原音の log-mag STFT, 原音の log-mag STFT)

使用例:
  python train_restorer.py "../cn-youtube-downloader/downloads/Amor, Fortuna.mp3" \\
      -o restorer.pt --epochs 200
"""

import argparse
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
CHUNK_N = int(2.0 * SR)   # 2秒チャンク = 88200 サンプル
STRIDE_N = CHUNK_N // 2   # 50% オーバーラップ
EPS = 1e-8
SHIFT_AMOUNTS = [4, 5, 6]  # 推論条件（−5半音）に近い量に絞る。両方向で合計 6 ラウンドトリップ


# ---------------------------------------------------------------------------
# データ生成ユーティリティ
# ---------------------------------------------------------------------------

def _rb_shift(y: np.ndarray, n: int) -> np.ndarray:
    """Rubber Band で n 半音シフト（モノラル）"""
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
    """波形 → log 振幅スペクトログラム (F, T) ※ F = 513"""
    _, _, Z = _stft(y, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    return np.log(np.abs(Z) + EPS).astype(np.float32)


def _pad8(spec: np.ndarray) -> np.ndarray:
    """時間軸を 8 の倍数にゼロパッド（U-Net の 3 段 pool に必要）"""
    T = spec.shape[1]
    T8 = ((T + 7) // 8) * 8
    return np.pad(spec, ((0, 0), (0, T8 - T))) if T8 > T else spec


def build_samples(y_clean: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    ラウンドトリップで (劣化版 log-mag, 原音 log-mag) チャンクのリストを返す。
    1 曲 × 8 ラウンドトリップ × ~150 チャンク = ~1200 サンプル。
    """
    samples: list[tuple[np.ndarray, np.ndarray]] = []

    for n in SHIFT_AMOUNTS:
        for sign in (+1, -1):
            print(f"  ラウンドトリップ: {sign*n:+d} → {-sign*n:+d} 半音 ...", flush=True)
            y_deg = _rb_shift(_rb_shift(y_clean, sign * n), -sign * n)
            wav_len = min(len(y_clean), len(y_deg))

            for start in range(0, wav_len - CHUNK_N, STRIDE_N):
                c_deg   = y_deg[start : start + CHUNK_N]
                c_clean = y_clean[start : start + CHUNK_N]
                samples.append((_pad8(_to_logmag(c_deg)), _pad8(_to_logmag(c_clean))))

    print(f"  → 合計 {len(samples)} チャンク（{CHUNK_N / SR:.0f}秒 × {len(samples)}）")
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray]], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        X, Y = self.samples[idx]
        if self.augment:
            # log 領域での加算 ≡ 波形への乗算ゲイン（−7〜0 dB 相当）
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
    # 音源読み込み（モノラル化）
    import subprocess, tempfile as _tf
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".mp3":
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", args.input]
        if args.end:
            from pitch_shift import parse_time
            ffmpeg_cmd += ["-t", str(parse_time(args.end))]
        with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(ffmpeg_cmd + ["-ar", "44100", "-f", "wav",
                           "-acodec", "pcm_f32le", tmp_path],
                           check=True, capture_output=True)
            y, sr = sf.read(tmp_path, dtype="float32")
        finally:
            os.unlink(tmp_path)
    else:
        y, sr = sf.read(args.input, dtype="float32")
        if args.end:
            from pitch_shift import parse_time
            y = y[:int(parse_time(args.end) * sr)]
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import soxr
        y = soxr.resample(y, sr, SR, quality="VHQ")
    print(f"音源: {len(y) / SR:.1f}秒 @ {SR} Hz")

    # 学習データ生成
    print("ラウンドトリップ学習データ生成中...")
    samples = build_samples(y)

    # Train / Val 分割（8:2）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    n_val = max(1, len(samples) // 5)
    tr_samples = [samples[i] for i in idx[n_val:]]
    va_samples = [samples[i] for i in idx[:n_val]]

    tr_dl = DataLoader(PairDataset(tr_samples, augment=True),
                       batch_size=args.batch, shuffle=True,  drop_last=True)
    va_dl = DataLoader(PairDataset(va_samples, augment=False),
                       batch_size=args.batch, shuffle=False)

    device = _get_device()
    model = PitchRestorer().to(device)

    if args.finetune:
        ck = torch.load(args.finetune, map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"Fine-Tune元: {args.finetune}  (val_loss={ck['val_loss']:.4f})")
        if args.lr == 3e-4:
            args.lr = 1e-5

    n_params = sum(p.numel() for p in model.parameters())
    print(f"デバイス: {device}  パラメータ数: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5, min_lr=1e-6
    )
    crit  = nn.L1Loss()

    best_val   = float("inf")   # チェックポイント用（生のval_loss）
    best_ema   = float("inf")   # 早期終了用（EMA平滑後）
    ema_val    = None
    no_improve = 0
    out_path   = os.path.join(SCRIPT_DIR, args.output)
    epoch_times: list[float] = []
    t_start = time.time()

    def _eta(remaining_epochs: int) -> str:
        if not epoch_times:
            return "算出中"
        avg = sum(epoch_times[-10:]) / len(epoch_times[-10:])
        secs = int(avg * remaining_epochs)
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}分{s:02d}秒"

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()

        # --- train ---
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

        # --- val ---
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, Y in va_dl:
                va_loss += crit(model(X.to(device)), Y.to(device)).item()
        va_loss /= len(va_dl)

        # EMA平滑化（α=0.2 ≒ 5epoch窓）
        ema_val = va_loss if ema_val is None else 0.8 * ema_val + 0.2 * va_loss

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
            marker = f"  ★EMA更新{ckpt_marker}"
        else:
            no_improve += 1
            eta = _eta(args.patience - no_improve)
            marker = (f"  [改善なし {no_improve:2d}/{args.patience}]"
                      f"  残り {eta}{ckpt_marker}")
            if no_improve >= args.patience:
                print(f"  Epoch {ep:4d}/{args.epochs}"
                      f"  train={tr_loss:.4f}  val={va_loss:.4f}"
                      f"  ema={ema_val:.4f}  lr={lr_now:.1e}{marker}")
                print(f"  早期終了 (EMAベース patience={args.patience}, epoch={ep})")
                break

        print(f"  Epoch {ep:4d}/{args.epochs}"
              f"  train={tr_loss:.4f}  val={va_loss:.4f}"
              f"  ema={ema_val:.4f}  lr={lr_now:.1e}{marker}", flush=True)

    elapsed = int(time.time() - t_start)
    h, r = divmod(elapsed, 3600)
    m, s = divmod(r, 60)
    print(f"完了。ベストモデル: {out_path}  (val_loss={best_val:.4f})"
          f"  経過時間: {h}:{m:02d}:{s:02d}")


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ピッチシフト劣化復元モデルの学習")
    ap.add_argument("input", help="学習用音源（MP3/WAV/FLAC）")
    ap.add_argument("-o", "--output", default="restorer.pt",
                    help="保存先モデルファイル (default: restorer.pt)")
    ap.add_argument("--epochs",   type=int,   default=200)
    ap.add_argument("--batch",    type=int,   default=16)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--patience", type=int,   default=60,
                    help="早期終了の patience エポック数 (default: 60)")
    ap.add_argument("-e", "--end",
                    help="学習に使う終端時刻（例: 2:31 または 151）。省略時は全体を使用。")
    ap.add_argument("--finetune", metavar="MODEL_PT",
                    help="事前学習済みモデルから Fine-Tune（LR を自動的に 1e-5 に変更）")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
