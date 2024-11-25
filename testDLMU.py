import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import natsort as ns
from sklearn.metrics import confusion_matrix
from MODELdl3plusmunet import MobileUNet

class SimplifiedCustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = ns.natsorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def class_to_mask(mask_class):
    colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

    mask_rgb = np.zeros((mask_class.shape[0], mask_class.shape[1], 3), dtype=np.uint8)
    for cls, color in colors.items():
        mask_rgb[mask_class == cls] = color
    return mask_rgb

def compute_iou(preds, gts, num_classes):
    hist = confusion_matrix(preds.flatten(), gts.flatten(), labels=range(num_classes))
    ious = []
    for cls in range(1, num_classes):
        intersection = hist[cls, cls]
        union = hist[cls, :].sum() + hist[:, cls].sum() - intersection
        ious.append(intersection / max(union, 1))
    return np.array(ious), np.nanmean(ious)

def load_model(device, model_path='teacher_epoch_200.pth', num_classes=4):
    model = MobileUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def test_and_evaluate(model, dataloader, output_dir, anno_dir, num_classes=4):
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for idx, (images) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds_batch = torch.argmax(outputs, dim=1).cpu().numpy()

            batch_img_names = dataloader.dataset.images[idx * dataloader.batch_size:(idx + 1) * dataloader.batch_size]

            for pred, img_name in zip(preds_batch, batch_img_names):
                pred_rgb = class_to_mask(pred)
                pred_img = Image.fromarray(pred_rgb)
                pred_img.save(os.path.join(output_dir, f"{img_name}.png"))
                preds.append(pred)


                gt_filename = os.path.splitext(img_name)[0] + ".png"
                gt_path = os.path.join(anno_dir, gt_filename)
                gt = np.array(Image.open(gt_path))
                gts.append(gt)

    class_iou, mIoU = compute_iou(np.concatenate(preds), np.concatenate(gts), num_classes)
    print(f"Class IoU: {class_iou}")
    print(f"Mean IoU: {mIoU:.4f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, '', num_classes=4)

    image_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])

    test_dataset = SimplifiedCustomDataset(
        image_dir='dataset/test/orgin',
        transform=image_transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    output_dir = 'dataset/test/result/DLMU'
    color_dir = 'dataset/test/color'
    test_and_evaluate(model, test_loader, output_dir, color_dir)

