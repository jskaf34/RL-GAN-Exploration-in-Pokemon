import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class GAN(): 
    def __init__(self, latent_size, image_size):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.latent_size = latent_size
        self.image_size = image_size

        self.discriminator = self.create_discriminator()
        self.discriminator.to(self.device)
        self.generator = self.create_generator(latent_size)


    def train_generator(self, optimizer_g, batch_size):
        self.discriminator.eval()
        self.generator.train()
        optimizer_g.zero_grad()
        
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        
        # Passing fake images as 1 through discriminator
        fake_images.to(self.device)
        preds = self.discriminator(fake_images)
        targets = torch.ones(batch_size, 1, device=self.device)
        loss = F.binary_cross_entropy(preds, targets)
        
        loss.backward()
        optimizer_g.step()
    
        return loss.item()

    def train_discriminator(self, real_images, optimizer_d, batch_size):
        self.generator.eval()
        self.discriminator.train()
        optimizer_d.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        loss = real_loss + fake_loss
        loss.backward()
        optimizer_d.step()
        return loss.item(), real_score, fake_score

    def train_gan(self, epochs, lr, train_dl):
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        for epoch in range(epochs):
            for real_images, _ in tqdm(train_dl):
                batch_size = real_images.shape[0]

                real_images = real_images.to(self.device)
                loss_d, real_score, fake_score = self.train_discriminator(real_images, optimizer_d, batch_size)

                loss_g = self.train_generator(optimizer_g, batch_size)
                
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
        return losses_g, losses_d, real_scores, fake_scores

    @staticmethod
    def create_generator(latent_size): 
        return nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )
    
    @staticmethod
    def create_discriminator(): 
        return nn.Sequential(
            # in: 3 x 64 x 64

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid()
        )

if __name__ == "__main__": 
    DATA_DIR = "./data"

    EPOCHS = 10
    LATENT_SIZE = 256
    LR = 1e-6

    IMAGE_SIZE = 64
    BATCH_SIZE = 16
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_ds = ImageFolder(DATA_DIR, transform=tt.Compose([ tt.Resize(IMAGE_SIZE),
                                                        tt.CenterCrop(IMAGE_SIZE),
                                                        tt.ToTensor(),
                                                        tt.Normalize(*stats)]))
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)

    gan = GAN(LATENT_SIZE, IMAGE_SIZE)
    history = gan.train_gan(
        epochs=EPOCHS, 
        lr=LR, 
        train_dl=train_dl
    )