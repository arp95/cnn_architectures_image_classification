# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
from vgg16 import *
import csv


# constants
model_path = "/home/arpitdec5/Desktop/cnn_architectures_image_classification/scripts/best_model.pth"
files = "/home/arpitdec5/Desktop/cnn_architectures_image_classification/data/"
result_file = "/home/arpitdec5/Desktop/cnn_architectures_image_classification/result.csv"
num_classes = 2
classes_map = {"0": "cat", "1": "dog"}

# define transforms and get data
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_data = torchvision.datasets.ImageFolder(files, transform=transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

# define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# eval model
results = []
results.append(["file_path", "prediction"])
model.eval()
with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, predicted = output.max(1)
        results.append([test_loader.dataset.samples[i][0], classes_map[str(int(predicted[0]))]])

# write results
with open(result_file, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for result in results:
        spamwriter.writerow(result)
