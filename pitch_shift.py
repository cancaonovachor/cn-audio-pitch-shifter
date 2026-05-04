#!/usr/bin/env python3
"""高音質ピッチシフター (Rubber Band Library + HPSS使用)"""

import argparse
import sys
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import librosa
from score_onset import build_score_onset_mask

PERFECT_FOURTH_DOWN = -5  # 完全4度 = 半音5つ下

_audiosr_model = None  # lazy-loaded singleton


def parse_time(s: str) -> float:
    """'MM:SS' または秒数文字列を秒のfloatに変換"""
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def read_audio(path: str, end: float | None = None) -> tuple[np.ndarray, int]:
    """MP3含む各種フォーマットをfloat32配列で読み込む。end指定時はその秒数でトリム。"""
    ext = os.path.splitext(path)[1].lower()
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", path]
    if end is not None:
        ffmpeg_cmd += ["-t", str(end)]
    ffmpeg_cmd += ["-ar", "44100", "-f", "wav", "-acodec", "pcm_f32le"]

    if ext == ".mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ffmpeg_cmd + [tmp_path],
                check=True,
                capture_output=True,
            )
            y, sr = sf.read(tmp_path, dtype="float32")
        finally:
            os.unlink(tmp_path)
    else:
        y, sr = sf.read(path, dtype="float32")
        if end is not None:
            y = y[:int(end * sr)]

    return y, sr


def rubberband(
    y: np.ndarray,
    sr: int,
    n_steps: int,
    formant: bool = True,
    oversample: int = 1,
) -> np.ndarray:
    """
    Rubber Band CLI でピッチシフト。

    oversample > 1 のとき、処理前に soxr_vhq で sr*oversample にアップサンプリングし、
    処理後に元の sr へダウンサンプリングする。
    高い周波数解像度でボコーダーが動くためエイリアシングと高域の滲みが減少する。
    """
    import soxr

    sr_proc = sr * oversample
    y_proc = soxr.resample(y, sr, sr_proc, quality="VHQ") if oversample > 1 else y

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
        in_path = fin.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
        out_path = fout.name

    try:
        sf.write(in_path, y_proc, sr_proc, subtype="FLOAT")
        cmd = ["rubberband", "--fine", "--pitch", str(n_steps)]
        if formant:
            cmd.append("--formant")
        cmd += [in_path, out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        y_shifted, _ = sf.read(out_path, dtype="float32")
    finally:
        os.unlink(in_path)
        os.unlink(out_path)

    if oversample > 1:
        y_shifted = soxr.resample(y_shifted, sr_proc, sr, quality="VHQ")

    return y_shifted.astype(np.float32)


def build_onset_mask(y_percussive: np.ndarray, sr: int, decay_ms: float) -> np.ndarray:
    """
    オンセット検出に基づく減衰マスクを生成する。
    各アタック位置で1.0に立ち上がり、decay_ms かけて指数的に0へ減衰する。
    持続音の区間はほぼ0になるため、percussive成分が倍音に干渉しない。
    """
    onset_samples = librosa.onset.onset_detect(y=y_percussive, sr=sr, units="samples")

    n = len(y_percussive)
    mask = np.zeros(n, dtype=np.float32)
    decay_n = int(decay_ms / 1000 * sr)
    # 5τ = decay_ms になるよう設定（τ経過で1/e ≈ 37%）
    decay_curve = np.exp(-np.arange(decay_n) / (decay_n / 5.0)).astype(np.float32)

    for s in onset_samples:
        end = min(s + decay_n, n)
        length = end - s
        mask[s:end] = np.maximum(mask[s:end], decay_curve[:length])

    # クリック防止: 1ms Gaussianスムージング
    return gaussian_filter1d(mask, sigma=sr * 0.001).astype(np.float32)


def hpss_pitch_shift(
    y: np.ndarray,
    sr: int,
    n_steps: int,
    decay_ms: float,
    use_onset_mask: bool = True,
    score_mask: np.ndarray | None = None,
    percussive_floor: float = 0.35,
    oversample: int = 1,
) -> np.ndarray:
    """
    HPSS + オンセットマスクによるピッチシフト。

    - Harmonic: 全区間に適用（倍音・持続音）。--formant で音色を保持。
    - Percussive: オンセットマスクを適用するが percussive_floor を下限とする。
      floor > 0 にすることで、持続音区間でも高域エネルギーが維持される。
    - Stereo: Mid = (L+R)/2、Side = L-R で分離し、復元は L = Mid + Side/2。
    """
    y_mono = y.mean(axis=1) if y.ndim == 2 else y

    print("  HPSS分離中...")
    y_harmonic, y_percussive = librosa.effects.hpss(y_mono, margin=3)

    print("  倍音成分をピッチシフト中（--formant あり）...")
    h_shifted = rubberband(y_harmonic, sr, n_steps, formant=True, oversample=oversample)

    print("  子音・打鍵成分をピッチシフト中（--formant なし）...")
    p_shifted = rubberband(y_percussive, sr, n_steps, formant=False, oversample=oversample)

    min_len = min(len(h_shifted), len(p_shifted))

    if score_mask is not None:
        raw_mask = np.resize(score_mask, len(p_shifted))
    elif use_onset_mask:
        print(f"  オンセットマスク生成中（信号ベース, 減衰 {decay_ms:.0f}ms）...")
        raw_mask = build_onset_mask(y_percussive, sr, decay_ms)
        raw_mask = np.resize(raw_mask, len(p_shifted))
    else:
        raw_mask = np.ones(len(p_shifted), dtype=np.float32)

    # 持続音区間でも percussive_floor 分の高域を保持する
    mask_with_floor = np.maximum(raw_mask, percussive_floor).astype(np.float32)
    y_result = h_shifted[:min_len] + p_shifted[:min_len] * mask_with_floor[:min_len]

    if y.ndim == 2:
        # Mid=(L+R)/2, Side=L-R → L = Mid + Side/2, R = Mid - Side/2
        y_side = y[:, 0] - y[:, 1]
        side_shifted = rubberband(y_side, sr, n_steps, formant=False, oversample=oversample)
        side_shifted = side_shifted[:min_len]
        l = y_result + side_shifted / 2
        r = y_result - side_shifted / 2
        y_result = np.stack([l, r], axis=1)

    return y_result.astype(np.float32)


def _patch_numpy_legacy() -> None:
    """audiosr 同梱の librosa 0.9.2 は NumPy 2.x で削除された np.float 等を使うため補完する"""
    for attr, val in [
        ('float', np.float64), ('complex', np.complex128),
        ('int', np.int64), ('bool', np.bool_),
    ]:
        if not hasattr(np, attr):
            setattr(np, attr, val)


def _get_torch_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ai_enhance(y: np.ndarray, sr: int, ddim_steps: int = 50) -> np.ndarray:
    """
    AudioSR (拡散モデルベース) で帯域を拡張・音質向上する。

    AudioSR の推奨最大入力長は 5.12 秒のため、長い音声は 5 秒チャンクに分割して
    順次処理し、境界部分を 20ms クロスフェードで繋ぎ合わせる。
    出力は 48kHz モノラルなので soxr VHQ で元の sr にダウンサンプリングして返す。
    ステレオ入力は各チャンネル個別に処理する。
    """
    _patch_numpy_legacy()
    import soxr
    import torch
    from audiosr import build_model, super_resolution  # type: ignore

    global _audiosr_model
    if _audiosr_model is None:
        device = _get_torch_device()
        print(f"  AudioSR モデルロード中 (device={device})...")
        _audiosr_model = build_model(model_name="basic", device=device)

    CHUNK_SEC = 5.0
    XFADE_MS = 20.0
    chunk_samples = int(CHUNK_SEC * sr)
    xfade_samples = int(XFADE_MS / 1000 * sr)

    def _process_chunk(ch_data: np.ndarray) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, ch_data, sr, subtype="PCM_24")
            out = super_resolution(
                _audiosr_model, tmp_path,
                seed=42, ddim_steps=ddim_steps, guidance_scale=3.5,
            )
        finally:
            os.unlink(tmp_path)
        # out shape: (1, 1, N) at 48 kHz
        out_mono = out.reshape(-1).astype(np.float32)
        result = soxr.resample(out_mono, 48000, sr, quality="VHQ").astype(np.float32)
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return result

    def _enhance_mono(ch_data: np.ndarray) -> np.ndarray:
        if len(ch_data) <= chunk_samples:
            return _process_chunk(ch_data)

        # 5秒チャンクに分割して順次処理
        chunks = [ch_data[i:i + chunk_samples]
                  for i in range(0, len(ch_data), chunk_samples)]
        n = len(chunks)
        processed = []
        for idx, chunk in enumerate(chunks):
            print(f"    チャンク {idx + 1}/{n} ({len(chunk) / sr:.1f}秒)...")
            processed.append(_process_chunk(chunk))

        # 20ms クロスフェードで結合してクリックを防ぐ
        result = processed[0]
        for nxt in processed[1:]:
            if len(result) < xfade_samples or len(nxt) < xfade_samples:
                result = np.concatenate([result, nxt])
                continue
            fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
            fade_in  = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)
            xfade = result[-xfade_samples:] * fade_out + nxt[:xfade_samples] * fade_in
            result = np.concatenate([result[:-xfade_samples], xfade, nxt[xfade_samples:]])
        return result

    n_chunks_est = int(np.ceil(len(y) / sr / CHUNK_SEC))
    print(f"  AI補完中 (ddim_steps={ddim_steps}, チャンク={CHUNK_SEC:.0f}秒 × 約{n_chunks_est}個)...")
    if y.ndim == 1:
        return _enhance_mono(y)

    channels = [_enhance_mono(y[:, ch]) for ch in range(y.shape[1])]
    min_len = min(len(c) for c in channels)
    return np.stack([c[:min_len] for c in channels], axis=1).astype(np.float32)


def apply_restorer(y: np.ndarray, sr: int, model_path: str) -> np.ndarray:
    """
    train_restorer.py で学習したモデルでピッチシフト劣化を復元する。

    STFT log-magnitude に対して残差補正を適用し、元の位相と組み合わせて
    波形を再構成する。ステレオ入力は各チャンネル個別に処理する。
    """
    from scipy.signal import stft as _stft, istft as _istft
    import torch
    from model_restorer import PitchRestorer, N_FFT, HOP_LENGTH

    device = _get_torch_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = PitchRestorer().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    EPS = 1e-8

    def _process_ch(ch_data: np.ndarray) -> np.ndarray:
        _, _, Z = _stft(ch_data, fs=sr, window="hann",
                        nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
        mag   = np.abs(Z).astype(np.float32)
        phase = np.angle(Z).astype(np.float32)
        log_mag = np.log(mag + EPS)

        F, T = log_mag.shape
        T8 = ((T + 7) // 8) * 8
        x_np = np.pad(log_mag, ((0, 0), (0, T8 - T))) if T8 > T else log_mag

        x = torch.from_numpy(x_np[None, None]).to(device)   # (1,1,F,T8)
        with torch.no_grad():
            y_hat = model(x).squeeze().cpu().numpy()          # (F,T8)
        y_hat = y_hat[:, :T]

        S = np.exp(y_hat) * np.exp(1j * phase)
        _, out = _istft(S, fs=sr, window="hann",
                        nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
        n = len(ch_data)
        if len(out) >= n:
            return out[:n].astype(np.float32)
        return np.pad(out, (0, n - len(out))).astype(np.float32)

    print(f"  劣化復元中（モデル: {os.path.basename(model_path)}）...")
    if y.ndim == 1:
        return _process_ch(y)

    channels = [_process_ch(y[:, ch]) for ch in range(y.shape[1])]
    min_len = min(len(c) for c in channels)
    return np.stack([c[:min_len] for c in channels], axis=1).astype(np.float32)


def high_shelf_sos(fc: float, gain_db: float, sr: int, q: float = 0.707) -> np.ndarray:
    """Audio EQ Cookbook に基づくハイシェルフフィルタ"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    two_sqrt_A_alpha = 2 * sqrt_A * alpha

    b0 =     A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =     A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
    a0 =         (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
    a1 =     2 * ((A - 1) - (A + 1) * cos_w0)
    a2 =         (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def peak_eq_sos(fc: float, gain_db: float, sr: int, q: float = 1.5) -> np.ndarray:
    """Audio EQ Cookbook に基づくピーキングEQフィルタ"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)

    b0 =  1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 =  1 - alpha / A

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def apply_eq(y: np.ndarray, sr: int, n_steps: int = -5) -> np.ndarray:
    """
    ピッチシフト後の音こもり補正EQ。
    シフト量に比例してゲインをスケールする（基準: -5半音 = 完全4度）。
    半音を下げるほど高域エネルギーが失われるため、補正量も大きくなる。
    """
    scale = abs(n_steps) / 5.0
    presence_gain = round(2.0 * scale, 1)   # 3kHz プレゼンス
    shelf_gain    = round(2.5 * scale, 1)   # 6kHz~ ハイシェルフ
    sos_all = np.vstack([
        peak_eq_sos(fc=3000, gain_db=presence_gain, sr=sr, q=1.5),
        high_shelf_sos(fc=6000, gain_db=shelf_gain,    sr=sr, q=0.707),
    ])
    if y.ndim == 1:
        return signal.sosfilt(sos_all, y).astype(np.float32)
    return np.stack(
        [signal.sosfilt(sos_all, y[:, ch]).astype(np.float32) for ch in range(y.shape[1])],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="高音質ピッチシフター（完全4度下げデフォルト）"
    )
    parser.add_argument("input", help="入力音声ファイル（MP3/WAV/FLACなど）")
    parser.add_argument(
        "-s", "--semitones",
        type=int,
        default=PERFECT_FOURTH_DOWN,
        help=f"半音単位のシフト量（デフォルト: {PERFECT_FOURTH_DOWN} = 完全4度下）",
    )
    parser.add_argument(
        "-o", "--output",
        help="出力ファイルパス（省略時: 入力ファイル名_shifted.flac）",
    )
    parser.add_argument(
        "-e", "--end",
        help="トリム終端（例: 2:31 または 151）",
    )
    parser.add_argument(
        "--no-hpss",
        action="store_true",
        help="HPSS分離を無効化する（高速だが子音が劣化しやすい）",
    )
    parser.add_argument(
        "--no-onset-mask",
        action="store_true",
        help="オンセットマスクを無効化する（HPSS分離のみ、子音を全区間ミックス）",
    )
    parser.add_argument(
        "--decay-ms",
        type=float,
        default=100.0,
        help="オンセットマスクの減衰時間 [ms]（デフォルト: 100）。"
             "短いほど持続音への干渉が減る。長いほど子音が長く保たれる。",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="オーバーサンプリング倍率（デフォルト: 2）。"
             "1=なし, 2=88.2kHz, 4=176.4kHz。大きいほど高品質・処理遅。",
    )
    parser.add_argument(
        "--percussive-floor",
        type=float,
        default=0.35,
        help="オンセットマスクの下限値 0.0〜1.0（デフォルト: 0.35）。"
             "0 だと持続音で高域が消え、1 だと常に全パーカッシブをミックス。",
    )
    parser.add_argument(
        "--score",
        metavar="MUSICXML",
        help="MusicXML ファイルを指定すると DTW アライメントによるスコア情報ベースの"
             "オンセット検出を使用する（--no-hpss 時は無効）",
    )
    parser.add_argument(
        "--no-eq",
        action="store_true",
        help="EQ補正を無効化する",
    )
    parser.add_argument(
        "--ai-enhance",
        action="store_true",
        help="AudioSR による AI 帯域拡張を適用する（処理時間が大幅に増加する）",
    )
    parser.add_argument(
        "--ai-steps",
        type=int,
        default=50,
        metavar="N",
        help="AudioSR の DDIM ステップ数（デフォルト: 50）。少ないほど速いが品質が落ちる。",
    )
    parser.add_argument(
        "--restore",
        metavar="MODEL_PT",
        help="train_restorer.py で学習した劣化復元モデルのパス (.pt)。"
             "EQ 補正の前に適用される。",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"エラー: ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_shifted.flac"

    end_sec = parse_time(args.end) if args.end else None

    print(f"読み込み中: {args.input}")
    if end_sec:
        print(f"  トリム: 0:00 〜 {args.end} ({end_sec:.1f}s)")
    y, sr = read_audio(args.input, end=end_sec)
    channels = 1 if y.ndim == 1 else y.shape[1]
    print(f"  サンプルレート: {sr} Hz, チャンネル: {channels}ch, 長さ: {len(y)/sr:.1f}s")

    # スコアベースのオンセットマスクを事前に生成
    score_mask = None
    if args.score and not args.no_hpss and not args.no_onset_mask:
        if not os.path.exists(args.score):
            print(f"エラー: MusicXML が見つかりません: {args.score}", file=sys.stderr)
            sys.exit(1)
        print(f"スコア情報ロード中: {args.score}")
        y_mono = y.mean(axis=1) if y.ndim == 2 else y
        score_mask = build_score_onset_mask(args.score, y_mono, sr, args.decay_ms)
        print(f"  → スコアベース オンセットマスク生成完了")

    os_label = f"{args.oversample}x ({sr * args.oversample // 1000}kHz)" if args.oversample > 1 else "なし"
    print(f"ピッチシフト中: {args.semitones:+d} 半音  オーバーサンプリング: {os_label}")
    if args.no_hpss:
        y_shifted = rubberband(y, sr, args.semitones, formant=True, oversample=args.oversample)
    else:
        if score_mask is not None:
            mode = "HPSS + スコアベース オンセットマスク（DTW アライメント）"
        elif not args.no_onset_mask:
            mode = "HPSS + 信号ベース オンセットマスク"
        else:
            mode = "HPSS（全区間ミックス）"
        print(f"  モード: {mode}")
        y_shifted = hpss_pitch_shift(
            y, sr, args.semitones,
            decay_ms=args.decay_ms,
            use_onset_mask=not args.no_onset_mask,
            score_mask=score_mask,
            percussive_floor=args.percussive_floor,
            oversample=args.oversample,
        )

    if args.restore:
        if not os.path.exists(args.restore):
            print(f"エラー: モデルファイルが見つかりません: {args.restore}", file=sys.stderr)
            sys.exit(1)
        y_shifted = apply_restorer(y_shifted, sr, args.restore)

    if not args.no_eq:
        scale = abs(args.semitones) / 5.0
        print(f"EQ補正中: プレゼンス +{2.0*scale:.1f}dB @3kHz, ハイシェルフ +{2.5*scale:.1f}dB @6kHz"
              f"  （{args.semitones:+d}半音基準で自動スケール）")
        y_shifted = apply_eq(y_shifted, sr, n_steps=args.semitones)

    if args.ai_enhance:
        dur_min = len(y_shifted) / sr / 60
        print(f"AI帯域拡張中: AudioSR (ddim_steps={args.ai_steps})"
              f"  ※推定処理時間 {dur_min * 1.1:.0f}〜{dur_min * 1.1 * 1.1:.0f} 分")
        y_shifted = ai_enhance(y_shifted, sr, ddim_steps=args.ai_steps)

    peak = np.max(np.abs(y_shifted))
    if peak > 1.0:
        y_shifted /= peak
        print(f"  ピーク正規化: {20 * np.log10(1.0 / peak):.1f} dB")

    print(f"書き出し中: {args.output}")
    sf.write(args.output, y_shifted, sr, format="FLAC", subtype="PCM_24")
    print("完了。")


if __name__ == "__main__":
    main()
