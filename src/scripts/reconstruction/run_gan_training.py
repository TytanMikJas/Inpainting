import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.vae.vae import VAE
from src.models.gan.discriminator import Discriminator
from src.models.baseline.encoder import Encoder
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from matplotlib import pyplot as plt
from tqdm import tqdm

N_EPOCHS_PRE = 0
N_EPOCHS = 250
gamma = 100

etl = ETLProcessor(
    kaggle_dataset="mahmudulhaqueshawon/cat-image",
    raw_dir="../data/raw_data",
    split_dir="../data/data_splits",
)
train_loader, val_loader, test_loader = etl.process()

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

def denormalize(img: torch.Tensor) -> torch.Tensor:
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)
vae = VAE(
    input_dim=3,
    hidden_dim=128,
    residual_hiddens=64,
    num_residual_layers=1,
    latent_dim=1024,
)

vae.load_state_dict(
    torch.load("models/reconstruction/vae-simclr/simclr_vae_layers_maskedFalse_tau0.001.pt", map_location='cuda')
)
vae = vae.cuda()
disc = Discriminator(input_dim=3).cuda()
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=5e-4)
for epoch in range(N_EPOCHS_PRE):
    disc.train()
    total_loss = 0.0
    total = 0
    for x, _ in tqdm(train_loader):
        x = x.cuda()
        x_false = torch.randn_like(x).cuda()   
        with torch.no_grad():
            x_recon = vae(x)['recon']
        y = disc(x)
        y_false = disc(x_false)
        y_recon = disc(x_recon)
        optimizer_disc.zero_grad()
        loss = F.binary_cross_entropy_with_logits(y, torch.ones_like(y, device=y.device)) + \
               F.binary_cross_entropy_with_logits(y_recon, torch.zeros_like(y, device=y.device)) * 0.5 + \
               F.binary_cross_entropy_with_logits(y_false, torch.zeros_like(y, device=y.device)) * 0.5
        loss.backward()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        optimizer_disc.step()
    # Validation loss
    disc.eval()
    val_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.cuda()
            x_val_false = torch.randn_like(x_val).cuda()
            x_val_recon = vae(x_val)['recon']
            y_val = disc(x_val)
            y_val_false = disc(x_val_false)
            y_val_recon = disc(x_val_recon)
            loss_val = F.binary_cross_entropy_with_logits(y_val, torch.ones_like(y_val, device=y_val.device)) + \
                       F.binary_cross_entropy_with_logits(y_val_recon, torch.zeros_like(y_val_recon, device=y_val_recon.device)) * 0.5 + \
                       F.binary_cross_entropy_with_logits(y_val_false, torch.zeros_like(y_val_false, device=y_val_false.device)) * 0.5
            val_loss += loss_val.item() * x_val.size(0)
            val_total += x_val.size(0)
    print(f"Epoch {epoch+1}, Loss: {total_loss / total:.4f} Val Loss: {val_loss / val_total:.4f}")
vae = VAE(
    input_dim=3,
    hidden_dim=128,
    residual_hiddens=64,
    num_residual_layers=1,
    latent_dim=1024,
)
vae = vae.cuda()
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter



timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"logs/vae_gan_experiment_{timestamp}")
vae.eval()
optimizer_vae = torch.optim.Adam(vae.parameters())
optimizer_dis = torch.optim.Adam(disc.parameters(), lr=5e-5)
with torch.no_grad():
    img = next(iter(val_loader))[0][0]
    plt.figure(figsize=(3, 3))
    plt.imshow(denormalize(img).permute(1,2,0))
    plt.show()

for i in range(N_EPOCHS):
    loss_sum_vae = 0
    loss_sum_dis = 0
    total = 0
    vae.eval()
    if i % 10 == 0:
        with torch.no_grad():
            recon = vae(img.unsqueeze(0).cuda())
            plt.figure(figsize=(3, 3))
            plt.imshow(denormalize(recon['recon'].cpu().detach().squeeze(0)).permute(1,2,0))
            plt.title(f'Epoch {i} - Recon')
            plt.show()
    vae.train()
    for x, _ in tqdm(train_loader):
        x = x.cuda()
        with torch.no_grad():
            mu, log_var = vae.encode(x)
        z = vae.reparameterize(mu, log_var)
        z_fake = torch.rand_like(z, device=z.device)

        vae_out = vae(x)
        recon = vae_out['recon']
        L_prior = vae_out['partial_loss']
        recon_fake = vae.decode(z_fake)

        d_recon_features = disc.features(recon)
        d_real_features = disc.features(x).detach()
        L_dis = loss_discriminator(disc, x, recon, recon_fake.detach())
        L_vae = L_prior + gamma * (F.mse_loss(x, recon)) - L_dis

        optimizer_vae.zero_grad()
        L_vae.backward()
        optimizer_vae.step()

        y_real = disc(x)
        y_recon = disc(recon.detach())
        y_fake = disc(recon_fake)

        real_labels = torch.ones_like(y_real)
        fake_labels = torch.zeros_like(y_recon)

        L_dis = loss_discriminator(disc, x, recon.detach(), recon_fake.detach())

        optimizer_dis.zero_grad()
        L_dis.backward()
        optimizer_dis.step()

        loss_sum_vae += L_vae.item()
        loss_sum_dis += L_dis.item()
        total += 1

    # --- VALIDATION ---
    vae.eval()
    disc.eval()
    val_loss_vae = 0.0
    val_loss_dis = 0.0
    val_total = 0
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.cuda()
            mu_val, log_var_val = vae.encode(x_val)
            z_val = vae.reparameterize(mu_val, log_var_val)
            z_fake_val = torch.rand_like(z_val, device=z_val.device)

            vae_out_val = vae(x_val)
            recon_val = vae_out_val['recon']
            L_prior_val = vae_out_val['partial_loss']
            recon_fake_val = vae.decode(z_fake_val)

            d_recon_features_val = disc.features(recon_val)
            d_real_features_val = disc.features(x_val).detach()
            L_dis_val = loss_discriminator(disc, x_val, recon_val, recon_fake_val.detach())
            L_vae_val = L_prior_val + gamma * (F.mse_loss(x_val, recon_val)) - L_dis_val

            val_loss_vae += L_vae_val.item()
            val_loss_dis += L_dis_val.item()
            val_total += 1

    print(f'Loss epoch {i} - vae {round(loss_sum_vae / total,5)} - dis {round(loss_sum_dis / total,5)} | val vae {round(val_loss_vae / val_total,5)} - val dis {round(val_loss_dis / val_total,5)}')
    writer.add_scalar("Loss/VAE", loss_sum_vae / total, i)
    writer.add_scalar("Loss/Discriminator", loss_sum_dis / total, i)
    writer.add_scalar("Val_Loss/VAE", val_loss_vae / val_total, i)
    writer.add_scalar("Val_Loss/Discriminator", val_loss_dis / val_total, i)
vae.eval()
img = next(iter(val_loader))[0][0]
plt.imshow(denormalize(img).permute(1,2,0))
plt.show()
recon = vae(img.unsqueeze(0).cuda())
plt.imshow(denormalize(recon['recon'].cpu().detach().squeeze(0)).permute(1,2,0))
plt.show()
from src.scripts.utils import get_test_loader, show_images, evaluate_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataloader = get_test_loader()
vae = VAE(
    input_dim=3,
    hidden_dim=128,
    residual_hiddens=64,
    num_residual_layers=2,
    latent_dim=256,
)
vae.load_state_dict(
    torch.load("vae.pth", map_location='cuda')
)
mask_img, clean_img = next(iter(test_dataloader))
mask_imgs = mask_img.to(device)
original_imgs = clean_img.to(device)

with torch.no_grad():
    reconstructed = vae(mask_imgs)

show_images(denormalize(original_imgs),denormalize( mask_imgs), denormalize(reconstructed["recon"]), n=12)
evaluate_model(vae, test_dataloader, device)