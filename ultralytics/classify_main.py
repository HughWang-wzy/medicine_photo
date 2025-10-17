from ultralytics import YOLO
import torch
import torch.nn.functional as F

model = YOLO("yolo11x-cls.yaml").load("yolo11x-cls.pt")  # build from YAML and transfer weights
results = model.train(cfg ='/home/hugh/newdisk/medicine_photo/default.yaml')