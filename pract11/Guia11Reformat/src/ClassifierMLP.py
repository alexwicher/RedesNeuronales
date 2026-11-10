from torch import nn


class ClassifierMLP(nn.Module):

    def __init__(self, p_dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)
