import torch
from model import SimpleModel

model = SimpleModel()
model.eval()

x = torch.randn(1, 4)

torch.onnx.export(
    model,
    x,
    "model.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported successfully")
