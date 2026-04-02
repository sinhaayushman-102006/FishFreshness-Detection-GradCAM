import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from utils import format_confidence, freshness_score

classes = ['Fresh', 'Medium', 'Spoiled']

model = timm.create_model('efficientnet_v2_s', pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier.in_features, 3)
)

model.load_state_dict(torch.load("model/best_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    conf = format_confidence(confidence)
    score = freshness_score(conf, label)

    return label, conf, score, image