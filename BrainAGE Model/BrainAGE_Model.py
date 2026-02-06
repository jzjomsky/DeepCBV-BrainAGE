"""
BrainAGE 3D CNN Models
======================

A VGG-style 3D convolutional architecture for brain age prediction using T1 MRI and AICBV data.

This file supports:
    - BrainAGE_T1_Model_Weights.pkl
    - BrainAGE_AICBV_Model_Weights.pkl

Usage:
    from BrainAGE_Model import AgeRegressor, load_model

Author: Jordan Jomsky
"""

import torch
import torch.nn as nn
import torch.optim as optim

class AgeRegressor(nn.Module):
    """
    3D Convolutional Neural Network for Brain Age estimation.

    Input shape:
        (batch, 1, 193, 229, 193)

    Output:
        Age estimate (float)
    """

    def __init__(self):
        super(AgeRegressor, self).__init__()

        self.block1 = self._make_block(1, 16)
        self.block2 = self._make_block(16, 32)
        self.block3 = self._make_block(32, 64)
        self.block4 = self._make_block(64, 128)
        self.block5 = self._make_block(128, 256)

        # NOTE: update this if input resolution changes
        self.flatten_size = 64512

        self.fc = nn.Linear(self.flatten_size, 1)

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filename, map_location=device):
    """
    Load weights for either T1 or AICBV version.

    Examples:
        model = load_model("BrainAGE_T1_Model_Weights.pkl")
        model = load_model("BrainAGE_AICBV_Model_Weights.pkl")
    """
    model = torch.load("/mnt/data/BrainAGE_T1_Model_Weights.pkl", map_location=device)
    model = model.to(device)
    model.eval()
    print(f"Loaded pretrained weights from {filename}")
    return model