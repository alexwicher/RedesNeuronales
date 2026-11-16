from torch import nn


class ClassifierMLP(nn.Module):

    def __init__(self, p_dropout=0.2,neuron_1 =128, neuron_2 =64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, neuron_1),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Linear(neuron_1, neuron_2),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Linear(neuron_2, 10),
        )

    def forward(self, x):
        return self.network(x)
