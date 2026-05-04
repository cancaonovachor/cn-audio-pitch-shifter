"""ピッチシフト劣化復元 U-Net モデル定義"""

import torch
import torch.nn as nn

N_FFT = 1024
HOP_LENGTH = 256


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PitchRestorer(nn.Module):
    """
    STFT log-magnitude の残差補正を予測する全畳み込み U-Net。

    周波数軸 (513 bin) は変えず、時間軸のみ 3 段階ダウンサンプリングするため
    推論時に任意長の音声を処理できる。
    入力: (B, 1, 513, T)  →  出力: (B, 1, 513, T)
    出力 = 入力 + 残差補正（モデルは補正量だけ学習）
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        self.enc1 = _Block(1,     c)
        self.enc2 = _Block(c,   2*c)
        self.enc3 = _Block(2*c, 4*c)
        self.pool = nn.MaxPool2d((1, 2))
        self.bot  = _Block(4*c, 8*c, dropout=0.15)

        self.up3  = nn.ConvTranspose2d(8*c, 4*c, (1, 2), stride=(1, 2))
        self.dec3 = _Block(8*c, 4*c)
        self.up2  = nn.ConvTranspose2d(4*c, 2*c, (1, 2), stride=(1, 2))
        self.dec2 = _Block(4*c, 2*c)
        self.up1  = nn.ConvTranspose2d(2*c,   c, (1, 2), stride=(1, 2))
        self.dec1 = _Block(2*c,   c)
        self.out  = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _crop(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """アップサンプル後のサイズをスキップ接続に合わせてクロップ"""
        return x[:, :, :ref.shape[2], :ref.shape[3]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))

        d3 = self.dec3(torch.cat([self._crop(self.up3(b),  e3), e3], 1))
        d2 = self.dec2(torch.cat([self._crop(self.up2(d3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._crop(self.up1(d2), e1), e1], 1))
        return x + self.out(d1)
