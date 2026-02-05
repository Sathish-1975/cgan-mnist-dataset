import torch
import torch.nn as nn
from models import Generator, Discriminator
from utils import get_data_loader, save_sample_images, save_model

def train(n_epochs, batch_size, lr, b1, b2, latent_dim, n_classes, img_shape, sample_interval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(n_classes, latent_dim, img_shape).to(device)
    discriminator = Discriminator(n_classes, img_shape).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    dataloader = get_data_loader(batch_size)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size,)).to(device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

        if epoch % sample_interval == 0:
            save_sample_images(generator, latent_dim, n_classes, epoch, device)
            save_model(generator, discriminator, epoch)

    return generator, discriminator
