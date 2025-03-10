import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class MultilabelClassifier(nn.Module):
    """
    Initialize the model architecture
    Exactly the same as the one used in the classifier
    """
    def __init__(self, n_features):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.imageClass = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=n_features)
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'class': self.imageClass(x)
        }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 3 values for RGB
])


def predict(modelPath, imagePath):
    """
    Load the trained house model and classify an image.
    """
    model = MultilabelClassifier(3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # **Force loading on CPU if CUDA is unavailable**
    checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    model.eval()

    # Load and preprocess image
    raw_img = Image.open(imagePath).convert("RGB")  # ✅ Ensure image is RGB
    single_img = data_transforms(raw_img)
    single_img = single_img.unsqueeze(0)

    outputs = model(single_img.to(device))
    res = 0
    for out in outputs:
        _, predicted = torch.max(outputs[out], 1)
        res = predicted.item()

    return res
