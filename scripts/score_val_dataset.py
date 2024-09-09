import json
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision.transforms as T
from tqdm import tqdm

from face_classification.models_registry import get_resnet

model = get_resnet("mobilenet_v2", 2)
checkpoint = torch.load("/data/ubuntu/Face_classification/experiment/checkpoints/ckpt_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

val_folder = Path("/data/ubuntu/Face_classification/data/unlabeled_val_img")
im_numbers = [int(p.stem) for p in val_folder.iterdir()]
model.eval()
image_names2labels = {}
labels = ["None"] * max(im_numbers)


for img_path in tqdm(val_folder.iterdir()):
    image = PIL.Image.open(img_path)
    image_number = img_path.stem
    image_name = img_path.name

    image_array = np.array(image).astype("uint8")
    image_tensor = T.ToTensor()(image_array)
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    label = torch.argmax(output).item()
    image_names2labels[image_name] = label
    labels[int(image_number) - 1] = label


with open("val_labels.json", "w") as f:
    json.dump(image_names2labels, f, indent=4)

with open("inference_labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")
