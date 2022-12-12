import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import sys

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from model import Net, EnsembleNet

import onnx
import onnxsim

import numpy as np
np.random.seed(123)

# +
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX")
    parser.add_argument("--weightpath", type=str, default="extract_frames_train_config.json")
    args = parser.parse_args()
    weight_path = args.weightpath
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = "convnext_tiny_in22k.fb_in22k"
    batch_size = 2
    image_size = 224
    
    
    model = Net(backbone)
    model.to(device)
    pretrained_dict = torch.load(weight_path)
    model.load_state_dict(pretrained_dict)
    model.eval()
    
    with torch.no_grad():
        img = torch.zeros(batch_size, image_size, image_size, 3).to(device)
        y = model(img)  # dry run
#             # print(time.time() - t1)

#         # ONNX export
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = os.path.join('onnx_export', "model.onnx")

        input_names=['input']
        output_names=['output']
        # opset 17 support layernorm for convnext
        torch.onnx.export(model, img, f, verbose=False, opset_version=17, 
            input_names=input_names,
            output_names=output_names,          # the ONNX version to export the model to
            # export_params=True,
            do_constant_folding=True,
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})

        onnx_model = onnx.load(f)
#         print("Generate simply onnx model")
        simplified_onnx_model, success = onnxsim.simplify(onnx_model)
#         # assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
        simplified_onnx_model_path =  os.path.join('onnx_export', "model.simplified.onnx")

        print(f'Generating {simplified_onnx_model_path} ...')
        onnx.save_model(simplified_onnx_model, simplified_onnx_model_path, save_as_external_data=True, all_tensors_to_one_file=False)
        print('done')

        from onnxconverter_common.float16 import convert_float_to_float16_model_path
        from onnxconverter_common import auto_mixed_precision

        input_onnx_model = simplified_onnx_model_path
        output_onnx_model = os.path.join('onnx_export', "model.fp16.simplified.onnx")

#         onnx_model = onnx.load(input_onnx_model)

        test_data = {"input": np.random.rand(batch_size, image_size, image_size, 3).astype(np.float32)-0.5}

        # Could also use rtol/atol attributes directly instead of this
        def validate(res1, res2):
            for r1, r2 in zip(res1, res2):
                if not np.allclose(r1, r2, rtol=1e-5, atol=1e-7):
                    return False
            return True
        if "efficientnet" in backbone:
            model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(simplified_onnx_model, test_data, validate, keep_io_types=True)
        else:
            model_fp16 = convert_float_to_float16_model_path(input_onnx_model,
                                    keep_io_types=True)
        onnx.save(model_fp16, output_onnx_model)
