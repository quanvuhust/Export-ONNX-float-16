# INSTALL
```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
pip install --pre timm
pip install -r requirements.txt
```
# EXPORT ONNX FP16
``` python convert_onnx.py --weightpath ... ```
