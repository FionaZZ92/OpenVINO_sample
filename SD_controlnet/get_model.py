from pathlib import Path
import torch
import argparse
from torch.onnx import _export as torch_onnx_export
from openvino.tools.mo import convert_model
from openvino.runtime import serialize
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-b', '--batch', type = int, default = 1, required = True,
                      help='Required. batch_size for solving single/multiple prompt->image generation.')
    args.add_argument('-sd','--sd_weights', type = str, default="", required = True,
                      help='Specify the path of stable diffusion model')
    args.add_argument('-lt','--lora_type', type = str, default="", required = False,
                      help='Specify the type of lora weights, you can choose "safetensors" or "bin"')
    args.add_argument('-lw', '--lora_weights', type = str, default="", required = False,
                      help='Add lora weights to Stable diffusion.')
    # fmt: on
    return parser.parse_args()

args = parse_args()
###covnert controlnet to IR
controlnet = ControlNetModel.from_pretrained("sd-controlnet-canny", torch_dtype=torch.float32)
inputs = {
    "sample": torch.randn((args.batch*2, 4, 64, 64)), 
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((args.batch*2,77,768)),
    "controlnet_cond": torch.randn((args.batch*2,3,512,512)) #batch=2
}
'''dynamic_names = {
    "sample": {0: "batch"},
    "encoder_hidden_states": {0: "batch", 1: "sequence"},
    "controlnet_cond": {0: "batch"},
}'''

CONTROLNET_ONNX_PATH = Path('controlnet-canny.onnx')
CONTROLNET_OV_PATH = CONTROLNET_ONNX_PATH.with_suffix('.xml')
controlnet.eval()
with torch.no_grad():
    down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)

controlnet_output_names = [f"down_block_res_sample_{i}" for i in range(len(down_block_res_samples))]
controlnet_output_names.append("mid_block_res_sample")

if not CONTROLNET_OV_PATH.exists():
    if not CONTROLNET_ONNX_PATH.exists():

        with torch.no_grad():
            torch_onnx_export(controlnet, inputs, CONTROLNET_ONNX_PATH, input_names=list(inputs),
                output_names=controlnet_output_names,onnx_shape_inference=False, #dynamic_axes=dynamic_names,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ov_ctrlnet = convert_model(CONTROLNET_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_ctrlnet,CONTROLNET_OV_PATH)
    del ov_ctrlnet
    print('ControlNet successfully converted to IR')
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")


###convert SD-Unet model to IR
pipe = StableDiffusionControlNetPipeline.from_pretrained(args.sd_weights, controlnet=controlnet)
if args.lora_type == "bin":
    pipe.unet.load_attn_procs(args.lora_weights)
elif args.lora_type == "safetensors":
    print("==make sure you already generate new SD model with lora by diffusers.scripts.convert_lora_safetensor_to_diffusers.py==")
else:
    print("==No lora==")
UNET_ONNX_PATH = Path('unet_controlnet/unet_controlnet.onnx')
UNET_OV_PATH = UNET_ONNX_PATH.parents[1] / 'unet_controlnet.xml'

if not UNET_OV_PATH.exists():
    if not UNET_ONNX_PATH.exists():
        UNET_ONNX_PATH.parent.mkdir(exist_ok=True)
        inputs.pop("controlnet_cond", None)
        inputs["down_block_additional_residuals"] = down_block_res_samples
        inputs["mid_block_additional_residual"] = mid_block_res_sample

        unet = pipe.unet
        unet.eval()

        input_names = ["sample", "timestep", "encoder_hidden_states", *controlnet_output_names]
        '''dynamic_names = {
            "sample": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "controlnet_cond": {0: "batch"},
        }'''

        with torch.no_grad():
            torch_onnx_export(unet, inputs, str(UNET_ONNX_PATH), #dynamic_axes=dynamic_names,
                input_names=input_names, output_names=["sample_out"], onnx_shape_inference=False, opset_version=15)
        del unet
    del pipe.unet
    ov_unet = convert_model(UNET_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_unet,UNET_OV_PATH)
    del ov_unet
    print('Unet successfully converted to IR')
else:
    del pipe.unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")

###convert SD-text_encoder model to IR
TEXT_ENCODER_ONNX_PATH = Path('text_encoder.onnx')
TEXT_ENCODER_OV_PATH = TEXT_ENCODER_ONNX_PATH.with_suffix('.xml')

def convert_encoder_onnx(text_encoder:torch.nn.Module, onnx_path:Path):
    if not onnx_path.exists():
        input_ids = torch.ones((args.batch, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # infer model, just to make sure that it works
            text_encoder(input_ids)
            # export model to ONNX format
            torch_onnx_export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names=['tokens'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                opset_version=14,  # onnx opset version for export
                onnx_shape_inference=False
            )
        print('Text Encoder successfully converted to ONNX')

if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder_onnx(pipe.text_encoder, TEXT_ENCODER_ONNX_PATH)
    ov_txten = convert_model(TEXT_ENCODER_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_txten,TEXT_ENCODER_OV_PATH)
    print('Text Encoder successfully converted to IR')
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")


###convert VAE model to IR
VAE_DECODER_ONNX_PATH = Path('vae_decoder.onnx')
VAE_DECODER_OV_PATH = VAE_DECODER_ONNX_PATH.with_suffix('.xml')

def convert_vae_decoder_onnx(vae: torch.nn.Module, onnx_path: Path):
    """
    Convert VAE model to ONNX, then IR format. 
    Function accepts pipeline, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for ONNX conversion via torch.export, 
    Parameters: 
        vae (torch.nn.Module): VAE model
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((args.batch, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(vae_decoder, latents, onnx_path, input_names=[
                              'latents'], output_names=['sample'])
        print('VAE decoder successfully converted to ONNX')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder_onnx(pipe.vae, VAE_DECODER_ONNX_PATH)
    ov_vae = convert_model(VAE_DECODER_ONNX_PATH, compress_to_fp16=True)
    serialize(ov_vae,VAE_DECODER_OV_PATH)
    print('VAE decoder successfully converted to IR')
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")
