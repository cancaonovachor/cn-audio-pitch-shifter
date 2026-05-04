"""
スコア情報に基づくオンセット検出モジュール。

MusicXML → DTW アライメント → 音声上のオンセット位置 + 子音種別 → オンセットマスク
"""

import xml.etree.ElementTree as ET
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

# イタリア語子音の種類別減衰時間 [ms]
# 破裂音: 短く鋭い立ち上がり / 摩擦音: 長めのノイズ / 鼻音・流音: 中程度
_DECAY_BY_CONSONANT: dict[str, float] = {
    'p': 50, 'b': 55, 't': 50, 'd': 55,  # 破裂音
    'c': 55, 'g': 55, 'q': 50, 'k': 50,  # 破裂音
    'f': 120, 'v': 100, 's': 130, 'z': 110,  # 摩擦音
    'm': 80, 'n': 80,                          # 鼻音
    'l': 75, 'r': 75,                          # 流音
}
_VOWELS = set('aeiouàáèéìíòóùú')


def _consonant_decay_ms(syllable: str) -> float:
    if not syllable:
        return 100.0
    first = syllable[0].lower()
    if first in _VOWELS:
        return 60.0
    return _DECAY_BY_CONSONANT.get(first, 100.0)


def _parse_musicxml(path: str) -> tuple[list[dict], float, int]:
    """
    MusicXML の全声部からノート情報を抽出する。

    Returns:
        notes: List of dicts with keys:
            onset_div, pitch_midi, syllable, syllabic,
            is_new_word, divisions, tempo_qpm
        tempo_qpm: final tempo in quarter notes per minute
        divisions: divisions per quarter note (first value)
    """
    STEP_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    tree = ET.parse(path)
    root = tree.getroot()

    notes: list[dict] = []
    global_tempo_qpm = 192.0
    global_divisions = 2

    for part in root.findall('part'):
        local_divisions = global_divisions
        local_tempo_qpm = global_tempo_qpm
        local_beats = 4
        local_beat_type = 4
        cumulative_div = 0

        for measure in part.findall('measure'):
            cursor = 0
            last_nochord_cursor = 0  # cursor at start of last non-chord note

            for elem in measure:
                tag = elem.tag

                if tag == 'attributes':
                    d = elem.find('divisions')
                    if d is not None:
                        local_divisions = int(d.text)
                        global_divisions = local_divisions
                    t = elem.find('time')
                    if t is not None:
                        b = t.findtext('beats')
                        bt = t.findtext('beat-type')
                        if b:
                            local_beats = int(b)
                        if bt:
                            local_beat_type = int(bt)

                elif tag == 'direction':
                    sound = elem.find('sound')
                    if sound is not None and 'tempo' in sound.attrib:
                        local_tempo_qpm = float(sound.attrib['tempo'])
                        global_tempo_qpm = local_tempo_qpm

                elif tag == 'note':
                    duration = int(elem.findtext('duration', '0') or '0')
                    is_rest = elem.find('rest') is not None
                    is_chord = elem.find('chord') is not None
                    is_grace = elem.find('grace') is not None
                    is_tie_stop = any(
                        t.get('type') == 'stop' for t in elem.findall('tie')
                    )

                    if is_grace:
                        continue

                    if is_chord:
                        note_onset = cumulative_div + last_nochord_cursor
                    else:
                        note_onset = cumulative_div + cursor
                        last_nochord_cursor = cursor
                        cursor += duration

                    if not is_rest and not is_tie_stop:
                        pitch_el = elem.find('pitch')
                        if pitch_el is not None:
                            step = pitch_el.findtext('step', 'C')
                            octave = int(pitch_el.findtext('octave', '4'))
                            alter = int(float(pitch_el.findtext('alter', '0') or 0))
                            midi = (octave + 1) * 12 + STEP_SEMITONE.get(step, 0) + alter

                            lyric_el = elem.find('lyric')
                            syllable = ''
                            syllabic = 'single'
                            if lyric_el is not None:
                                syllable = lyric_el.findtext('text', '').strip()
                                syllabic = lyric_el.findtext('syllabic', 'single').strip()

                            notes.append({
                                'onset_div': note_onset,
                                'pitch_midi': midi,
                                'syllable': syllable,
                                'syllabic': syllabic,
                                'is_new_word': syllabic in ('begin', 'single'),
                                'divisions': local_divisions,
                                'tempo_qpm': local_tempo_qpm,
                            })

                elif tag == 'forward':
                    cursor += int(elem.findtext('duration', '0') or '0')
                elif tag == 'backup':
                    cursor -= int(elem.findtext('duration', '0') or '0')

            # 小節の長さを拍子記号から計算して次小節先頭へ進める
            measure_dur = int(local_beats * (4.0 / local_beat_type) * local_divisions)
            cumulative_div += measure_dur

    return notes, global_tempo_qpm, global_divisions


def _onset_to_sec(onset_div: int, divisions: int, tempo_qpm: float) -> float:
    return onset_div / divisions * 60.0 / tempo_qpm


def _build_score_chroma(
    notes: list[dict],
    sr: int,
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    """ノートのMIDIピッチからスコアクロマ行列（12 × n_frames）を構築する"""
    chroma = np.zeros((12, n_frames), dtype=np.float32)

    for note in notes:
        onset_sec = _onset_to_sec(note['onset_div'], note['divisions'], note['tempo_qpm'])
        # 持続時間は近似値（1拍分）で十分 - クロマの重なりが重要
        beat_sec = 60.0 / note['tempo_qpm']
        f0 = int(onset_sec * sr / hop_length)
        f1 = int((onset_sec + beat_sec) * sr / hop_length)
        pc = note['pitch_midi'] % 12
        f0 = max(0, f0)
        f1 = min(n_frames, f1)
        if f0 < f1:
            chroma[pc, f0:f1] += 1.0

    col_max = chroma.max(axis=0, keepdims=True)
    chroma = np.divide(chroma, col_max, out=np.zeros_like(chroma), where=col_max > 0)
    return chroma


def _build_warp_fn(score_chroma: np.ndarray, audio_chroma: np.ndarray):
    """
    DTW でスコアフレーム → 音声フレームのマッピング関数を返す。
    score_chroma, audio_chroma: (12, n_frames)
    """
    # NaN・inf を除去してから DTW へ渡す
    score_chroma = np.nan_to_num(score_chroma, nan=0.0, posinf=0.0)
    audio_chroma = np.nan_to_num(audio_chroma, nan=0.0, posinf=0.0)

    # 零ベクトルの列は cosine が未定義になるため微小値を加算
    score_chroma += 1e-8
    audio_chroma += 1e-8

    _, wp = librosa.sequence.dtw(
        X=score_chroma, Y=audio_chroma, metric='cosine', backtrack=True
    )
    # wp shape: (N, 2), wp[i] = [score_frame, audio_frame]
    score_frames = wp[:, 0]
    audio_frames = wp[:, 1]

    n_score = int(score_frames.max()) + 1
    lookup = np.zeros(n_score, dtype=np.float32)
    for sf in range(n_score):
        matched = audio_frames[score_frames == sf]
        lookup[sf] = float(np.median(matched)) if len(matched) > 0 else (lookup[sf - 1] if sf > 0 else 0)

    def warp(score_frame: float) -> float:
        return float(lookup[int(np.clip(score_frame, 0, n_score - 1))])

    return warp


def build_score_onset_mask(
    musicxml_path: str,
    y_audio: np.ndarray,
    sr: int,
    default_decay_ms: float = 100.0,
) -> np.ndarray:
    """
    MusicXML と音声を DTW アライメントし、スコア情報に基づくオンセットマスクを返す。

    信号ベース検出と異なり、楽譜上の全声部のノート発音位置を把握したうえで
    DTW で実時間にマッピングするため、ポリフォニーでの検出精度が向上する。
    また歌詞の子音種別から decay_ms を自動決定する。

    Args:
        musicxml_path: MusicXML ファイルパス
        y_audio: モノラル音声配列
        sr: サンプルレート
        default_decay_ms: 歌詞なしノートのデフォルト減衰時間 [ms]
    Returns:
        onset_mask: shape (len(y_audio),) の float32 配列
    """
    HOP = 512

    print("  MusicXML 解析中...")
    notes, tempo_qpm, divisions = _parse_musicxml(musicxml_path)
    print(f"    → {len(notes)} notes, tempo={tempo_qpm:.0f} QPM, div/quarter={divisions}")

    # スコアの最終ノートから全体フレーム数を推定
    max_onset_sec = max(
        _onset_to_sec(n['onset_div'], n['divisions'], n['tempo_qpm']) for n in notes
    )
    n_score_frames = int((max_onset_sec + 5.0) * sr / HOP)
    n_audio_frames = int(np.ceil(len(y_audio) / HOP))

    print("  スコアクロマ構築中...")
    score_chroma = _build_score_chroma(notes, sr, HOP, n_score_frames)

    print("  音声クロマ計算中...")
    audio_chroma = librosa.feature.chroma_cqt(y=y_audio, sr=sr, hop_length=HOP)
    audio_chroma = audio_chroma[:, :n_audio_frames]

    # スコアを音声長の1.25倍に制限してDTW計算コストを抑える
    score_chroma_trim = score_chroma[:, : n_audio_frames + n_audio_frames // 4]

    print("  DTW アライメント中...")
    warp = _build_warp_fn(score_chroma_trim, audio_chroma)

    print("  オンセットマスク構築中...")
    mask = np.zeros(len(y_audio), dtype=np.float32)

    for note in notes:
        syllabic = note['syllabic']
        syllable = note['syllable']
        is_new_word = note['is_new_word']

        # 語頭 (begin/single): 子音種別から decay_ms を決定
        # 語中・語末 (middle/end): 発音変化は弱いので短め
        if syllable:
            decay_ms = _consonant_decay_ms(syllable) if is_new_word else 35.0
        else:
            decay_ms = default_decay_ms * 0.5 if not is_new_word else default_decay_ms

        # スコアフレーム → 音声フレーム → サンプル位置
        score_frame = _onset_to_sec(note['onset_div'], note['divisions'], note['tempo_qpm']) * sr / HOP
        audio_frame = warp(score_frame)
        onset_sample = int(audio_frame * HOP)

        decay_n = int(decay_ms / 1000 * sr)
        if decay_n < 1:
            continue
        tau = decay_n / 5.0
        decay_curve = np.exp(-np.arange(decay_n) / tau).astype(np.float32)

        s = max(0, onset_sample)
        e = min(len(mask), onset_sample + decay_n)
        if s < e:
            mask[s:e] = np.maximum(mask[s:e], decay_curve[: e - s])

    mask = gaussian_filter1d(mask, sigma=sr * 0.001).astype(np.float32)
    return mask
