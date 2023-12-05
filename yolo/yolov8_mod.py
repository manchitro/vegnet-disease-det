# %%
from ultralytics import YOLO
import hiddenlayer as hl
import os

print(os.getcwd())
model_name = 'yolov8l'
exp_name = 'original_from_pt'
model = YOLO('yolov8.yaml')
model.load(f'{model_name}.pt')  # build from YAML and transfer weights
# print(model)

results = model.train(
	data = './vegnet_yolo.yaml',
	epochs = 1000,
	batch = 8,
	imgsz = 256,
	device = 0,
	workers = 16,
	optimizer = 'Adam',
	pretrained = False,
	val = True,
	plots = True,
	save = True,
	save_period = 10,
	show = True,
	patience = 50,
	lr0 = 0.00001,
	lrf = 0.001,
	fliplr = 0.0,
	amp = False,
	exist_ok = True,
	name = f'{model_name}_{exp_name}',
	project = '../vegnet_yolo_out'
)

print(results)
path = model.export(format="onnx")  # export the model to ONNX format
print(path)


