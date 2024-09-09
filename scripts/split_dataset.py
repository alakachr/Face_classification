import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

images_paths = [p for p in Path("/data/ubuntu/Face_classification/data/images").glob("*")]
indexes = range(0, len(images_paths))
with open("/data/ubuntu/Face_classification/data/labels.txt", "r") as f:
    labels = f.readlines()
print(len(images_paths), len(labels))
print(labels[6976], images_paths[6976].name)
print(labels[73], images_paths[73].name)
print(labels[36], images_paths[36].name)
val_folder = "/data/ubuntu/Face_classification/data/val_img"

indexes_train, indexes_val = train_test_split(indexes, test_size=0.2, random_state=42)
im_name2label_train = {}
im_name2label_val = {}

for i in tqdm(indexes_train):
    image_path = images_paths[i]
    im_name2label_train[image_path.name] = int(labels[int(image_path.name.split(".")[0]) - 1])

for i in tqdm(indexes_val):
    image_path = images_paths[i]
    im_name2label_val[image_path.name] = int(labels[int(image_path.name.split(".")[0]) - 1])

with open("/data/ubuntu/Face_classification/data/im_name2label_train.json", "w") as f:
    json.dump(im_name2label_train, f, indent=4)
with open("/data/ubuntu/Face_classification/data/im_name2label_val.json", "w") as f:
    json.dump(im_name2label_val, f, indent=4)
