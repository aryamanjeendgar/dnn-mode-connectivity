import torch.nn as nn
import curves

class MLP_base(nn.Module):
    def __init__(self, num_classes: int):
        super(MLP_base, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class MLP_curve(nn.Module):
    def __init__(self, num_classes: int, fix_points: list[bool]):
        super(MLP_curve, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            curves.Linear(28 * 28, 128, fix_points=fix_points),
            nn.ReLU(),
            curves.Linear(128, 64, fix_points=fix_points),
            nn.ReLU(),
            curves.Linear(64, num_classes, fix_points=fix_points)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class MLP:
    base = MLP_base
    curve = MLP_curve
    kwargs = {}
