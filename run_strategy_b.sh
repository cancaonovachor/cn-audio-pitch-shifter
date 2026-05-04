#!/bin/bash
# 戦略B自動実行スクリプト
# 戦略Aの完了を待ってから事前学習 → Fine-Tuneを順番に実行する

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
DOWNLOADS="$SCRIPT_DIR/../cn-youtube-downloader/downloads"
LOG="$SCRIPT_DIR/strategy_b.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# 他のスクリプトのインスタンスが動いていたら待つ
check_and_wait_existing() {
    local timeout=30
    while pgrep -f "pretrain_restorer|train_restorer" >/dev/null 2>&1; do
        if [ $timeout -le 0 ]; then
            log "警告: 既存プロセスがタイムアウト。続行します。"
            break
        fi
        log "既存の学習プロセスが実行中。待機中... ($timeout秒)"
        sleep 5
        timeout=$((timeout - 5))
    done
}

check_and_wait_existing

# 戦略Aの完了待ち
STRATEGY_A_PID=$1
if [ -n "$STRATEGY_A_PID" ]; then
    log "戦略A（PID: $STRATEGY_A_PID）の完了を待機中..."
    while kill -0 "$STRATEGY_A_PID" 2>/dev/null; do
        sleep 30
    done
    log "戦略A完了。戦略Bを開始します。"
else
    log "PID未指定。すぐに戦略Bを開始します。"
fi

# Phase 1: 事前学習（25曲）
log "=== Phase 1: 事前学習開始（25曲） ==="
"$PYTHON" "$SCRIPT_DIR/pretrain_restorer.py" \
    "$DOWNLOADS" \
    -o "$SCRIPT_DIR/pretrained.pt" \
    --epochs 200 \
    --batch 64 \
    --patience 25 \
    >> "$LOG" 2>&1
log "=== Phase 1 完了 ==="

# Phase 2: Fine-Tune（Amor Fortunaのみ）
log "=== Phase 2: Fine-Tune開始（Amor Fortuna） ==="
"$PYTHON" "$SCRIPT_DIR/train_restorer.py" \
    "$DOWNLOADS/Amor, Fortuna.mp3" \
    --finetune "$SCRIPT_DIR/pretrained.pt" \
    -o "$SCRIPT_DIR/restorer.pt" \
    --end 2:31 \
    --epochs 100 \
    --patience 40 \
    >> "$LOG" 2>&1
log "=== Phase 2 完了 ==="

log "戦略B完了。restorer.pt が更新されました。"
