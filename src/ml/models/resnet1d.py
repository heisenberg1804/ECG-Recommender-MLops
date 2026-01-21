# ============================================================
# FILE: src/ml/models/resnet1d.py
# ============================================================
"""
1D ResNet for ECG classification.

Adapted from torchvision ResNet for 1D signals.
Input: (batch, 12, 5000) - 12 leads, 5000 samples at 500Hz
Output: (batch, num_classes) - multi-label logits
"""

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Basic residual block for 1D signals."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet for ECG signals."""

    def __init__(
        self,
        layers: list[int],
        num_classes: int,
        in_channels: int = 12,
        include_patient_context: bool = False,
        patient_context_dim: int = 3,  # age, sex, bmi
    ):
        super().__init__()
        self.include_patient_context = include_patient_context
        self.in_channels = 64

        # Stem - initial convolution
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        fc_input_dim = 512
        if include_patient_context:
            fc_input_dim += patient_context_dim

        self.fc = nn.Linear(fc_input_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            BasicBlock1D(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, patient_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch, 12, 5000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 512)

        # Concatenate patient context if provided
        if self.include_patient_context and patient_context is not None:
            x = torch.cat([x, patient_context], dim=1)

        x = self.fc(x)  # (batch, num_classes)

        return x


def resnet18_1d(num_classes: int, include_patient_context: bool = False) -> ResNet1D:
    """Constructs a ResNet-18 1D model."""
    return ResNet1D(
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        include_patient_context=include_patient_context,
    )


def resnet34_1d(num_classes: int, include_patient_context: bool = False) -> ResNet1D:
    """Constructs a ResNet-34 1D model."""
    return ResNet1D(
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        include_patient_context=include_patient_context,
    )


if __name__ == "__main__":
    # Test the model
    model = resnet18_1d(num_classes=5)
    x = torch.randn(2, 12, 5000)  # batch=2, 12 leads, 5000 samples
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
