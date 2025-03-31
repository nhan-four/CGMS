import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), -1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    def __init__(self, output_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        disc_input = torch.cat((data, labels), -1)
        return self.model(disc_input)

def train_cg(X_train, y_train, latent_dim=100, batch_size=128, epochs=300, lr_g=0.0004, lr_d=0.0001):

    minority_indices = np.where(y_train == 1)[0]
    data_minority = X_train[minority_indices]
    label_conditions = y_train[minority_indices]

    output_dim = X_train.shape[1]
    num_classes = 2  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, output_dim, num_classes).to(device)
    discriminator = Discriminator(output_dim, num_classes).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    label_conditions_one_hot = torch.eye(num_classes)[label_conditions].float().to(device)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        idx = np.random.randint(0, data_minority.shape[0], batch_size)
        real_data = torch.tensor(data_minority[idx, :], dtype=torch.float32).to(device)
        real_conditions = label_conditions_one_hot[idx]

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise, real_conditions)

        optimizer_D.zero_grad()
        d_loss_real = adversarial_loss(discriminator(real_data, real_conditions), real_labels)
        d_loss_fake = adversarial_loss(discriminator(fake_data.detach(), real_conditions), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_data, real_conditions), real_labels)
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    return generator

def generate_data(generator, X_train, y_train, latent_dim=100):
    num_samples_to_generate = len(X_train[y_train == 0]) - len(X_train[y_train == 1])
    noise = torch.randn(num_samples_to_generate, latent_dim).to(generator.device)
    condition_minority = torch.eye(2)[1].repeat(num_samples_to_generate, 1).to(generator.device)

    with torch.no_grad():
        generated_minority_data = generator(noise, condition_minority).cpu().numpy()

    X_train_augmented = np.vstack((X_train, generated_minority_data))
    y_train_augmented = np.hstack((y_train, np.ones(num_samples_to_generate)))

    print("Data balanced:")
    print(f"Class 0: {np.sum(y_train_augmented == 0)}")
    print(f"Class 1: {np.sum(y_train_augmented == 1)}")

    return X_train_augmented, y_train_augmented