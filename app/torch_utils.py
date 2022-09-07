import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image

# load model
PATH = "app/classifier.pth"
model=torch.load(PATH)
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze_(0)

# predict
def get_prediction(image_tensor):
    # images = image_tensor.reshape(-1, 28*28)
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
