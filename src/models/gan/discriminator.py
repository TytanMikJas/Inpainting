import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 3):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),  
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        out = self.fc_layers(x)
        return out
    
def loss_discriminator(disc, x, recon, recon_fake):
    y_real = disc(x)
    y_recon = disc(recon)
    y_fake = disc(recon_fake)

    real_labels = torch.ones_like(y_real)
    fake_labels = torch.zeros_like(y_recon)

    L_dis = (
        F.binary_cross_entropy_with_logits(y_real, real_labels) +
        (F.binary_cross_entropy_with_logits(y_recon, fake_labels)
        + F.binary_cross_entropy_with_logits(y_fake, fake_labels)) / 2
    )
    return L_dis