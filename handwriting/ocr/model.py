"""Lightweight CNN + BiLSTM + CTC model for handwriting OCR.

Architecture
------------
* Small CNN encoder (3 conv-pool blocks) that maps a fixed-height line image
  to a sequence of feature vectors.
* Bidirectional LSTM over the time axis.
* Linear head projecting to (alphabet_size + 1) logits (the +1 is the CTC
  blank token).

Design choices for low-VRAM hardware (GT 1030 / ~2 GB)
--------------------------------------------------------
* Image height is fixed at 64 px; width is variable (padded in batches).
* Small channel counts (32 → 64 → 128) keep the model under ~5 MB.
* No large pre-trained backbone.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional-Recurrent Neural Network for CTC-based sequence recognition.

    Parameters
    ----------
    alphabet:
        Ordered list of characters the model should recognise.  Index 0 is
        reserved for the CTC blank token internally (not in this list).
    img_height:
        Fixed height every input image will be resized to.
    cnn_channels:
        Tuple of output channel counts for the three CNN blocks.
    lstm_hidden:
        Hidden size of each LSTM direction.
    lstm_layers:
        Number of stacked BiLSTM layers.
    """

    def __init__(
        self,
        alphabet: List[str],
        img_height: int = 64,
        cnn_channels: tuple = (32, 64, 128),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.alphabet = alphabet
        self.blank_idx = 0  # CTC blank is index 0
        num_classes = len(alphabet) + 1  # +1 for blank

        c1, c2, c3 = cnn_channels

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
            # Block 2
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4
            # Block 3
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W unchanged
        )

        # After 3 pool layers: height becomes img_height // 8
        cnn_out_h = img_height // 8
        rnn_input_size = c3 * cnn_out_h

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.linear = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Float tensor of shape ``(B, 1, H, W)`` with values in [0, 1].

        Returns
        -------
        Log-softmax tensor of shape ``(T, B, num_classes)`` where
        ``T = W`` (one time step per input column after pooling).
        """
        # CNN feature map: (B, C, H', W')
        feat = self.cnn(x)

        b, c, h, w = feat.shape
        # Reshape to (T=W, B, C*H')
        feat = feat.permute(3, 0, 1, 2).reshape(w, b, c * h)

        out, _ = self.rnn(feat)        # (T, B, 2*hidden)
        logits = self.linear(out)      # (T, B, num_classes)
        return torch.log_softmax(logits, dim=2)


def build_model(
    alphabet: List[str],
    img_height: int = 64,
    device: str | None = None,
) -> CRNN:
    """Instantiate a :class:`CRNN` and move it to *device*."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN(alphabet=alphabet, img_height=img_height)
    return model.to(device)
