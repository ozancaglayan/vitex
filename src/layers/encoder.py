from collections import OrderedDict
from torch import nn


class VitexCNN(nn.Module):
    def __init__(self, n_blocks: int = 1, patch_width: int = 30,
                 patch_height: int = 24, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout: float = 0.0,
                 emb_size: int = 512):
        super().__init__()

        def _create_block():
            return nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)),
                ('bn', nn.BatchNorm2d(1, affine=False)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p=dropout)),
            ]))

        self.blocks = nn.ModuleList([_create_block() for i in range(n_blocks)])
        self.proj = nn.Linear(patch_width * patch_height, emb_size)

    def forward(self, x):
        x = x.float().unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.proj(x.view((x.shape[0], -1)))
