import enum
from os import error
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dcgan_mnist import Generator
from dcgan_mnist import Discriminator

writer = SummaryWriter("./logs/")

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_dir = "./data"
image_size = 64
batch_size = 64

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, 100, 1, 1, device=device)

real_label = 1.
fake_label = 0.

d_optimizer = optim.Adam(discriminator.parameters(), 0.0002, (0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), 0.0002, (0.5, 0.999))

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

discriminator.apply(init_weights)
generator.apply(init_weights)

def train(dataloader, epoch) -> None:
    batches = len(dataloader)
    for i, data in enumerate(dataloader, 0):
        discriminator.zero_grad()

        real_batch = data[0].to(device)
        batch_size = real_batch.size(0)
        label = torch.full((batch_size, 1, 1, 1), real_label, dtype=torch.float, device=device)
        output = discriminator(real_batch)
        errorD_real = criterion(output, label)
        errorD_real.backward()
        d_real = output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_batch = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_batch.detach())
        errorD_fake = criterion(output, label)
        errorD_fake.backward()
        D_G_z1 = output.mean().item()

        errorD = errorD_fake + errorD_real

        d_optimizer.step()

        generator.zero_grad()
        real_batch = data[0].to(device)
        batch_size = real_batch.size(0)
        label.fill_(real_label)
        output = discriminator(fake_batch)
        errorG = criterion(output, label)
        errorG.backward()
        D_G_z2 = output.mean().item()

        g_optimizer.step()

        iters = i + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", errorD.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", errorG.item(), iters)
        writer.add_scalar("Train_Adversarial/D_Real", d_real, iters)
        writer.add_scalar("Train_Adversarial/D_Fake1", D_G_z1, iters)
        writer.add_scalar("Train_Adversarial/D_Fake2", D_G_z2, iters)
        # Print the loss function every ten iterations and the last iteration in this epoch.
        if (i + 1) % 10 == 0 or (i + 1) == batches:
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/5]({i + 1:05d}/{batches:05d}) "
                  f"D Loss: {errorD.item():.6f} G Loss: {errorG.item():.6f} "
                  f"D(Real): {d_real:.6f} D(Fake1)/D(Fake2): {D_G_z1:.6f}/{D_G_z2:.6f}.")


def main() -> None:
    dataset = torchvision.datasets.MNIST(root=dataset_dir,
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize([image_size, image_size]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, ], [0.5, ]),
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)
    for epoch in range(5):
        train(dataloader, epoch)

        torch.save(discriminator.state_dict(), f"./d_epoch{epoch + 1}.pth")
        torch.save(generator.state_dict(), f"./g_epoch{epoch + 1}.pth")

main()
if __name__ == "__main__":
    main()