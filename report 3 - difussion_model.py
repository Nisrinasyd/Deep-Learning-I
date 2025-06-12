!pip install torch torchvision matplotlib numpy sympy==1.12

# === 1. Import and Setup ===
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === 2. Load FashionMNIST Dataset ===
transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.FashionMNIST(
    root="./fashionmnist", train=True, download=True, transform=transform
)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# take first batch 
data_iter = iter(train_loader)
x, y = next(data_iter)

# Visualization
plt.figure(figsize=(8, 2))
plt.imshow(torchvision.utils.make_grid(x[:8], nrow=8).permute(1, 2, 0).squeeze(), cmap="gray")
plt.axis("off")
plt.title("FashionMNIST Samples")
plt.show()



#Corruption Function (Gaussian Noise)
def corrupt(x, amount):
    """Add noise Gaussian with scale 'amount'."""
    amount = amount.view(-1, 1, 1, 1)
    noise = torch.randn_like(x)
    return x * (1 - amount) + noise * amount, noise


#Simple UNet
class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        ])
        self.act = nn.SiLU()
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:
                h.append(x)
                x = self.down(x)
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.up(x)
                x += h.pop()
            x = self.act(layer(x))
        return x

#Training Loop
net = BasicUNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
n_epochs = 5
losses = []

for epoch in range(n_epochs):
    for x, _ in train_loader:
        x = x.to(device)
        noise_amount = torch.rand(x.size(0), device=device)
        x_noised, noise = corrupt(x, noise_amount)
        
        pred_noise = net(x_noised)
        loss = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses[-len(train_loader):]) / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f}")


#Visualization and Denoising Result
x, _ = next(iter(train_loader))
x = x[:8].to(device)
amount = torch.linspace(0, 1, x.shape[0], device=device).view(-1, 1, 1, 1)
x_noised, noise = corrupt(x, amount)

with torch.no_grad():
    pred_noise = net(x_noised)
    x_denoised = (x_noised - pred_noise).clamp(0, 1).cpu()

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
axs[0].set_title("Original Images")
axs[0].imshow(torchvision.utils.make_grid(x.cpu())[0], cmap='Greys')

axs[1].set_title("Noisy Inputs")
axs[1].imshow(torchvision.utils.make_grid(x_noised.cpu())[0], cmap='Greys')

axs[2].set_title("Denoised Output")
axs[2].imshow(torchvision.utils.make_grid(x_denoised)[0], cmap='Greys')
plt.tight_layout()
plt.show()


