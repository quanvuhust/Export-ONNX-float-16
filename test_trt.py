import tensorrt as trt
import numpy as np
import sys
# from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import time
trt.init_libnvinfer_plugins(None,'')
TRT_LOGGER = trt.Logger()
 
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()
 
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
 
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
 
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
 
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
 
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, batch_size, image_size, bindings, inputs, outputs, stream):
    (IN_IMAGE_H, IN_IMAGE_W) = image_size
    context.set_binding_shape(0, (batch_size, IN_IMAGE_H, IN_IMAGE_W, 3))
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
 
    context.execute_v2(bindings=bindings)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
 
engine_path = 'model.trt'
image_size = (224, 224)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    print(f)
    engine = runtime.deserialize_cuda_engine(f.read())
    print(engine)
    context = engine.create_execution_context()
    max_batch_size = 32
    buffers = allocate_buffers(engine, max_batch_size)
 
    inputs, outputs, bindings, stream = buffers
#     print('Length of inputs: ', len(inputs))
    batch_size = 32
    x = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
    for i in range(1000):
        t0 = time.time()
        # time.sleep(1)
        # inputs[0].host = x
        trt_outputs = do_inference(context, batch_size, image_size, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(time.time() - t0)
