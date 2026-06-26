import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession("model.onnx")

input_data = {"input": np.random.randn(1, 4).astype(np.float32)}

start = time.time()
output = session.run(None, input_data)
end = time.time()

print("Output:", output)
print("Latency:", end - start)
