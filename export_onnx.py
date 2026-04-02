import torch
from model import model

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, dummy, "model/model.onnx")
print("ONNX exported!")