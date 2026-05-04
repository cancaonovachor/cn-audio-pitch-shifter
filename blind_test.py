#!/usr/bin/env python3
"""
ブラインドA/Bテストスクリプト
v1_os2x と v1_os2x_restored をランダムな区間で聴き比べ

使い方:
  # クリップ生成（5箇所・各10秒）
  python blind_test.py compare_XXX/

  # 答えを表示
  python blind_test.py compare_XXX/ --reveal

  # 回答と正解率を表示
  python blind_test.py compare_XXX/ --reveal --score 1:A 2:B 3:A 4:B 5:A
"""

import argparse
import json
import os
import random
import subprocess
import sys


LABEL_A = "v1_os2x"
LABEL_B = "v1_os2x_restored"


def find_files(compare_dir: str) -> tuple[str, str]:
    files = {}
    for fname in os.listdir(compare_dir):
        if not fname.endswith(".flac"):
            continue
        path = os.path.join(compare_dir, fname)
        if "v1_os2x_restored" in fname:
            files["restored"] = path
        elif fname.endswith("_v1_os2x.flac"):
            files["os2x"] = path
    if "os2x" not in files or "restored" not in files:
        print("エラー: v1_os2x.flac / v1_os2x_restored.flac が見つかりません", file=sys.stderr)
        sys.exit(1)
    return files["os2x"], files["restored"]


def get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_clip(src: str, start: float, duration: float, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ss", str(start), "-t", str(duration),
         "-c:a", "flac", dst],
        capture_output=True, check=True,
    )


def generate(compare_dir: str, n: int, duration: float, margin: float,
             output_dir: str, seed: int) -> None:
    os2x_path, restored_path = find_files(compare_dir)
    total = get_duration(os2x_path)

    max_start = total - duration - margin
    if max_start <= margin:
        print(f"エラー: 音源が短すぎます（{total:.1f}秒）", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    starts = sorted(rng.uniform(margin, max_start) for _ in range(n))

    os.makedirs(output_dir, exist_ok=True)

    answers = {}
    print(f"\nクリップ生成中（各{duration:.0f}秒 × {n}箇所）...")
    for i, start in enumerate(starts, 1):
        restored_is_A = rng.choice([True, False])
        answers[str(i)] = {
            "A": "restored" if restored_is_A else "os2x",
            "B": "os2x" if restored_is_A else "restored",
            "start_sec": round(start, 1),
        }
        a_src = restored_path if restored_is_A else os2x_path
        b_src = os2x_path if restored_is_A else restored_path
        extract_clip(a_src, start, duration, os.path.join(output_dir, f"clip_{i:02d}_A.flac"))
        extract_clip(b_src, start, duration, os.path.join(output_dir, f"clip_{i:02d}_B.flac"))
        mm, ss = divmod(int(start), 60)
        print(f"  クリップ{i:02d}: {mm}:{ss:02d} 〜 ({duration:.0f}秒)")

    key_path = os.path.join(output_dir, "answers.json")
    with open(key_path, "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n出力先: {output_dir}/")
    print(f"答えは answers.json に保存済み（--reveal で表示）\n")
    print_questions(n)


def print_questions(n: int) -> None:
    print("=" * 48)
    print("  ブラインドテスト  （A/B どちらが良い？）")
    print("=" * 48)
    for i in range(1, n + 1):
        print(f"  クリップ {i:02d}:  A ／ B  （どちらが良いですか？）")
    print("=" * 48)
    print("\n回答後に以下のコマンドで結果を表示:")
    print("  python blind_test.py <dir> --reveal --score 1:A 2:B ...")


def reveal(output_dir: str, scores: list[str]) -> None:
    key_path = os.path.join(output_dir, "answers.json")
    if not os.path.exists(key_path):
        print("エラー: answers.json が見つかりません。先に生成してください。", file=sys.stderr)
        sys.exit(1)
    with open(key_path) as f:
        answers = json.load(f)

    user = {}
    for s in scores:
        parts = s.split(":")
        if len(parts) == 2:
            user[parts[0].strip()] = parts[1].strip().upper()

    print("\n" + "=" * 48)
    print("  答え合わせ")
    print("=" * 48)

    correct = 0
    for i_str, info in sorted(answers.items(), key=lambda x: int(x[0])):
        mm, ss = divmod(int(info["start_sec"]), 60)
        a_label = "restored（復元モデル）" if info["A"] == "restored" else "os2x（元）"
        b_label = "restored（復元モデル）" if info["B"] == "restored" else "os2x（元）"
        print(f"\n  クリップ {int(i_str):02d}  [{mm}:{ss:02d}〜]")
        print(f"    A = {a_label}")
        print(f"    B = {b_label}")

        if i_str in user:
            chosen = user[i_str]
            chosen_label = info[chosen] if chosen in info else "?"
            better = "restored" if chosen_label == "restored" else "os2x"
            is_correct = chosen_label == "restored"
            if is_correct:
                correct += 1
            print(f"    あなたの回答: {chosen} → {a_label if chosen=='A' else b_label}  {'✓ 復元モデル選択' if is_correct else '✗ os2x選択'}")

    if user:
        print(f"\n復元モデルを選んだ回数: {correct} / {len(user)}")
        if correct > len(user) / 2:
            print("→ 復元モデルの改善が聴感上でも確認できました")
        elif correct == len(user) / 2:
            print("→ 差がほぼ感じられない結果でした")
        else:
            print("→ os2x の方が好まれました（復元モデルが逆効果の可能性）")
    print("=" * 48)


def main() -> None:
    ap = argparse.ArgumentParser(description="ブラインドA/Bテスト生成・採点")
    ap.add_argument("compare_dir", help="compare_XXX ディレクトリ（生成時）またはblind_testディレクトリ（reveal時）")
    ap.add_argument("-n", "--clips", type=int, default=5, help="クリップ数 (default: 5)")
    ap.add_argument("-d", "--duration", type=float, default=10.0, help="各クリップの長さ[秒] (default: 10)")
    ap.add_argument("-m", "--margin", type=float, default=10.0, help="曲頭・曲末の除外[秒] (default: 10)")
    ap.add_argument("-o", "--output", default="blind_test", help="出力ディレクトリ (default: blind_test)")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")
    ap.add_argument("--reveal", action="store_true", help="答えを表示")
    ap.add_argument("--score", nargs="*", metavar="N:X",
                    help="回答を入力して正解率表示（例: --score 1:A 2:B 3:A）")
    args = ap.parse_args()

    if args.reveal:
        reveal(args.output, args.score or [])
    else:
        generate(args.compare_dir, args.clips, args.duration,
                 args.margin, args.output, args.seed)


if __name__ == "__main__":
    main()
