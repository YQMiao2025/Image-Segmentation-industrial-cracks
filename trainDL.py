import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from MODELdeeplabv3plus import get_model


num_epochs = 200
learning_rate = 0.0001
batch_size = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_flag = True


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, data_fraction=1.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


image_dir = 'dataset/train/origin'
mask_dir = 'dataset/train/mask'


transform_image = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform_image)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

weights = [0.05, 0.55, 0.05, 0.35]
weights = torch.tensor(weights).to(device)

num_classes = 4
model = get_model(num_classes)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 每100个epoch将学习率降低为原来的0.5倍


def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

                if debug_flag:
                    display_debug(images, masks, outputs)

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')

        if (epoch + 1) % 10 == 0:
            model_path = f'DeepLabV3plus_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Model saved as {model_path}')

    torch.save(model.state_dict(), 'DeepLabV3plus.pth')
    print('Model saved as DeepLabV3plus.pth')

def display_debug(images, masks, outputs):
    images = images.cpu().detach()
    masks = masks.cpu().detach()
    outputs = outputs.cpu().detach()

    for i in range(min(4, images.size(0))):
        image = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()
        output = torch.argmax(outputs[i], dim=0).numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='nipy_spectral')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(output, cmap='nipy_spectral')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.show(block=False)
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    train()