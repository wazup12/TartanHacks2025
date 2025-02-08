import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

filename = "test1.jpg"  

model_type = "DPT_Hybrid"  

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

grad_x, grad_y = np.gradient(output)

gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

incline_angle = np.degrees(np.arctan(gradient_magnitude))
print(incline_angle)   

np.save("gradient.npy", incline_angle)

print("Incline angle map saved as 'gradient.npy'")