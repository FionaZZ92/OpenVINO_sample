from pathlib import Path
import torch
import gc
import argparse
from typing import Tuple
from torch.onnx import _export as torch_onnx_export
import openvino as ov
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from functools import partial

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-b', '--batch', type = int, default = 1, required = True,
                      help='Required. batch_size for solving single/multiple prompt->image generation.')
    args.add_argument('-ctrl', '--controlnet', type = str, default = "", required = True,
                      help='Specify path of controlnet Model path')
    args.add_argument('-sd','--sd_weights', type = str, default="", required = True,
                      help='Specify the path of stable diffusion model')
    args.add_argument('-lt','--lora_type', type = str, default="", required = False,
                      help='Specify the type of lora weights, you can choose "safetensors" or "bin"')
    args.add_argument('-lw', '--lora_weights', type = str, default="", required = False,
                      help='Add lora weights to Stable diffusion.')
    # fmt: on
    return parser.parse_args()
adapter_id = "lcm-lora-sdv1-5"
TEXT_ENCODER_OV_PATH = Path('model/text_encoder.xml')
UNET_OV_PATH = Path('model/unet_controlnet.xml')
CONTROLNET_OV_PATH = Path('model/controlnet-canny.xml')
VAE_DECODER_OV_PATH = Path('model/vae_decoder.xml')
TOKENIZER_PATH = Path('model/tokenizer')
SCHEDULER_PATH = Path('model/scheduler')

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64
}

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def flattenize_inputs(inputs):
    """
    Helper function for resolve nested input structure (e.g. lists or tuples of tensors)
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

def prepare_input_info(input_dict):
    """
    Helper function for preparing input info (shapes and data types) for conversion based on example inputs
    """
    flatten_inputs = flattenize_inputs(inputs.values())
    input_info = []
    for input_data in flatten_inputs:
        updated_shape = list(input_data.shape)
        if updated_shape:
            updated_shape[0] = -1
        if input_data.ndim == 4:
            updated_shape[2] = -1
            updated_shape[3] = -1

        input_info.append((dtype_mapping[input_data.dtype], updated_shape))
    return input_info


args = parse_args()
###covnert controlnet to IR
controlnet = ControlNetModel.from_pretrained(args.controlnet, trust_remote_code=True, torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(args.sd_weights, controlnet=controlnet, trust_remote_code=True)
if args.lora_weights != "":
    pipe.load_lora_weights(adapter_id)
    # fuse LoRA weights with UNet
    pipe.fuse_lora()
text_encoder = pipe.text_encoder
text_encoder.eval()
unet = pipe.unet
unet.eval()
vae = pipe.vae
vae.eval()
del pipe
gc.collect()


inputs = {
    "sample": torch.randn((args.batch*2, 4, 64, 64)),
    "timestep": torch.tensor(1, dtype=torch.float32),
    "encoder_hidden_states": torch.randn((args.batch*2,77,768)),
    "controlnet_cond": torch.randn((args.batch*2,3,512,512))
}


# Prepare conditional inputs for U-Net
if not UNET_OV_PATH.exists():    
    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
    
if not CONTROLNET_OV_PATH.exists():
    input_info = prepare_input_info(inputs)
    with torch.no_grad():
        controlnet.forward = partial(controlnet.forward, return_dict=False)
        ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
        ov.save_model(ov_model, CONTROLNET_OV_PATH, compress_to_fp16=True)
        del ov_model
        cleanup_torchscript_cache()
    print('ControlNet successfully converted to IR')
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")

del controlnet
gc.collect()

#=======Convert unet==============

class UnetWrapper(torch.nn.Module):
    def __init__(
        self, 
        unet, 
        sample_dtype=torch.float32, 
        timestep_dtype=torch.int64, 
        encoder_hidden_states=torch.float32, 
        down_block_additional_residuals=torch.float32, 
        mid_block_additional_residual=torch.float32
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self, 
        sample:torch.Tensor, 
        timestep:torch.Tensor, 
        encoder_hidden_states:torch.Tensor, 
        down_block_additional_residuals:Tuple[torch.Tensor],  
        mid_block_additional_residual:torch.Tensor
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample, 
            timestep, 
            encoder_hidden_states, 
            down_block_additional_residuals=down_block_additional_residuals, 
            mid_block_additional_residual=mid_block_additional_residual
        )

if not UNET_OV_PATH.exists():
    inputs.pop("controlnet_cond", None)
    inputs["down_block_additional_residuals"] = down_block_res_samples
    inputs["mid_block_additional_residual"] = mid_block_res_sample
    input_info = prepare_input_info(inputs)

    wrapped_unet = UnetWrapper(unet)
    wrapped_unet.eval()

    with torch.no_grad():
        ov_model = ov.convert_model(wrapped_unet, example_input=inputs)
        
    for (input_dtype, input_shape), input_tensor in zip(input_info, ov_model.inputs):
        input_tensor.get_node().set_partial_shape(ov.PartialShape(input_shape))
        input_tensor.get_node().set_element_type(input_dtype)
    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, UNET_OV_PATH, compress_to_fp16=True)
    del ov_model
    cleanup_torchscript_cache()
    del wrapped_unet
    del unet
    gc.collect()
    print('Unet successfully converted to IR')
else:
    del unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")
gc.collect()

#===========convert text_encoder=========
def convert_encoder(text_encoder:torch.nn.Module, ir_path:Path):
    """
    Convert Text Encoder model to OpenVINO IR. 
    Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
    Parameters: 
        text_encoder (torch.nn.Module): text_encoder model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((args.batch, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # inputs for model tracing
                input=([args.batch,77],)
            )
            ov.save_model(ov_model, ir_path, compress_to_fp16=True)
            del ov_model
        cleanup_torchscript_cache()
        print('Text Encoder successfully converted to IR')
    

if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
del text_encoder
gc.collect()

#==========convert vae============
def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model to IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for convert, 
    Parameters: 
        vae (torch.nn.Module): VAE model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(vae_decoder, example_input=latents, input=[args.batch, 4, 64, 64])
            ov.save_model(ov_model, ir_path, compress_to_fp16=True)
        del ov_model
        cleanup_torchscript_cache()
        print('VAE decoder successfully converted to IR')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

del vae
