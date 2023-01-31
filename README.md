# INSTALL
```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
pip install --pre timm
pip install -r requirements.txt
```
# EXPORT ONNX FP16
Convert pytorch model to onnx and quantize onnx model from float32 to float16. Support convnext and efficientnet, dynamic batch size

``` python convert_onnx.py --weightpath ... ```

# CONVERT TENSORRT FP16
CUDA_VISIBLE_DEVICES=0 ./trtexec --onnx=model.simplified.onnx --minShapes=input:1x224x224x3 --optShapes=input:24x224x224x3 --maxShapes=input:32x224x224x3 --workspace=2048 --saveEngine=model.fp16.trt --fp16
