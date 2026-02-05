from train import train

if __name__ == "__main__":
    hyperparams = {
        "n_epochs": 20,
        "batch_size": 128,
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.999,
        "latent_dim": 100,
        "n_classes": 10,
        "img_shape": (1, 28, 28),
        "sample_interval": 5
    }

    print("Starting CGAN training on MNIST...")
    generator, discriminator = train(**hyperparams)
    print("Training finished!")
