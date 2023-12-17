# %%
from ultralytics import YOLO
import os

print(os.getcwd())
model_name = 'yolov8s'
exp_name = 'original_trained_lr_0.001'
model = YOLO(model_name)
model.load(f'{model_name}.pt')  # build from YAML and transfer weights
# print(model)

results = model.train(
	data = './vegnet_yolo.yaml',
	epochs = 1000,
	batch = 32,
	imgsz = 256,
	device = 0,
	workers = 16,
	optimizer = 'Adam',
	pretrained = True,
	val = True,
	plots = True,
	save = True,
	save_period = 10,
	show = True,
	patience = 50,
	lr0 = 0.001,
	lrf = 0.01,
	fliplr = 0.0,
	seed=42,
	amp = False,
	exist_ok = True,
	name = f'{model_name}_{exp_name}',
	project = '../vegnet_yolo_out'
)

print(results)
# path = model.export(format="onnx")  # export the model to ONNX format
# print(path)


