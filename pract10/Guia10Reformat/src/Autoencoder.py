from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, latent_dim=64, p_dropout=0.0):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, latent_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        x_hat = self.decoder(z)

        x_hat = x_hat.view(
            x.size(0),
            1,
            28,
            28
        )

        return x_hat

    def encode(self, x):

        return self.encoder(x)

    def decode(self, z):

        x_hat = self.decoder(z)

        return x_hat.view(
            z.size(0),
            1,
            28,
            28
        )