#!/bin/bash
# 3バージョンのピッチシフトを一括生成して聴き比べ用ファイルを出力する

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
MAIN="$SCRIPT_DIR/pitch_shift.py"

INPUT="${1:?使い方: ./compare.sh <入力ファイル> [--end MM:SS]}"
shift

OUT_DIR="$SCRIPT_DIR/compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

BASE="$(basename "${INPUT%.*}")"

run() {
    local label="$1"; shift
    local out="$OUT_DIR/${BASE}_${label}.flac"
    echo ""
    echo "━━━ $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$PYTHON" "$MAIN" "$INPUT" -o "$out" "$@"
}

# v1: HPSS なし（位相ボコーダー単体 + EQ）、オーバーサンプリングなし
run "v1_basic"          --no-hpss --oversample 1 "$@"

# v1_os2x: v1 + 2x オーバーサンプリング（88.2kHz処理）
run "v1_os2x"           --no-hpss --oversample 2 "$@"

# v1_os2x_ai: v1_os2x + AudioSR AI帯域拡張
run "v1_os2x_ai"        --no-hpss --oversample 2 --ai-enhance "$@"

# v1_os4x: v1 + 4x オーバーサンプリング（176.4kHz処理）
run "v1_os4x"           --no-hpss --oversample 4 "$@"

# v2: HPSS あり、オンセットマスク なし（全区間で子音ミックス）
run "v2_hpss"         --no-onset-mask "$@"

# v3: HPSS あり、オンセットマスク あり（アタック部のみ子音を復元）
run "v3_hpss_onset"   "$@"

# v1_os2x_restored: v1_os2x + ラウンドトリップ学習済み劣化復元モデル
MODEL_FILE="$(dirname "$0")/restorer.pt"
if [ -f "$MODEL_FILE" ]; then
    run "v1_os2x_restored" --no-hpss --oversample 2 --restore "$MODEL_FILE" "$@"
else
    echo "restorer.pt が見つかりません。先に train_restorer.py を実行してください。"
fi

# v4: HPSS + スコアベース オンセットマスク（MusicXML + DTW アライメント）
SCORE_FILE="$(dirname "$0")/Amor, Fortuna.musicxml"
if [ -f "$SCORE_FILE" ]; then
    run "v4_score_dtw" --score "$SCORE_FILE" "$@"
else
    echo "MusicXML not found, skipping v4"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "出力先: $OUT_DIR"
ls -lh "$OUT_DIR"
