import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mnist_train = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        mnist_train, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return loader

def save_sample_images(generator, latent_dim, n_classes, epoch, device):
    generator.eval()
    n_rows = n_classes
    z = torch.randn(n_rows**2, latent_dim).to(device)
    # Get labels 0, 1, 2, ..., 9 repeated for each row
    labels = torch.LongTensor(np.array([num for _ in range(n_rows) for num in range(n_rows)])).to(device)
    
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    
    gen_imgs = 0.5 * gen_imgs + 0.5 # denormalize
    gen_imgs = gen_imgs.cpu().numpy()
    
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(10, 10))
    cnt = 0
    for i in range(n_rows):
        for j in range(n_rows):
            axs[i, j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    
    plt.savefig(f"images_epoch_{epoch}.png")
    plt.close()
    generator.train()

def save_model(generator, discriminator, epoch):
    torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")
