from PIL import Image
from diffusers import UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionControlNetPipeline, ControlNetModel
import torch
import numpy as np
import argparse
from typing import Union, List, Optional, Tuple
from diffusers.utils import load_image
from diffusers.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer
from openvino.runtime import Core, Model, Type
from openvino.runtime.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from safetensors.torch import load_file
import time
import cv2

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-lp', '--lora_path', type = str, default = "", required = False,
                      help='Specify path of lora weights *.safetensors')
    args.add_argument('-a','--alpha',type = float, default = 0.75, required = False,
                      help='Specify the merging ratio of lora weights, default is 0.75.')
    args.add_argument('-lp2', '--lora_path2', type = str, default = "", required = False,
                      help='Specify path of lora weights *.safetensors')
    args.add_argument('-a2','--alpha2',type = float, default = 0.75, required = False,
                      help='Specify the merging ratio of lora weights, default is 0.75.')                  
    return parser.parse_args()

def scale_fit_to_window(dst_width:int, dst_height:int, image_width:int, image_height:int):
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)

def preprocess(image: Image.Image):
    src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height), resample=Image.Resampling.LANCZOS))[None, :]
    pad_width = 512 - dst_width
    pad_height = 512 - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    #image = np.squeeze(image)
    #image = cv2.copyMakeBorder(image, int(pad_height//2), 512-int(pad_height//2)-dst_height, int(pad_width//2), 512-int(pad_width//2)-dst_width, cv2.BORDER_CONSTANT, (0,0,0) );
    #cv2.imwrite("preprocess.png",image)
    #image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    return image, pad


def randn_tensor(
    shape: Union[Tuple, List],
    dtype: Optional[np.dtype] = np.float32,
):
    latents = np.random.randn(*shape).astype(dtype)

    return latents

class InsertLoRA(MatcherPass):
    def __init__(self,lora_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Convert")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()
            for y in lora_dict_list:
                if root.get_friendly_name().replace('.','_').replace('_weight','') == y["name"]:
                    consumers = root_output.get_target_inputs()
                    lora_weights = ops.constant(y["value"],Type.f32,name=y["name"])
                    add_lora = ops.add(root,lora_weights,auto_broadcast='numpy')
                    for consumer in consumers:
                        consumer.replace_source_output(add_lora.output(0))

                    # For testing purpose
                    self.model_changed = True
                    # Use new operation for additional matching
                    self.register_new_node(add_lora)

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param,"InsertLoRA"), callback)


class OVContrlNetStableDiffusionPipeline(DiffusionPipeline):
    """
    OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
    """
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler,
        core: Core,
        controlnet: Model,
        text_encoder: Model,
        unet: Model,
        vae_decoder: Model,
        state_dict,
        alpha_list,
        device:str = "AUTO"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8 #2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scheduler = scheduler
        self.load_models(core, device, controlnet, text_encoder, unet, vae_decoder, state_dict, alpha_list)
        self.set_progress_bar_config(disable=True)
    

    def load_models(self, core: Core, device: str, controlnet:Model, text_encoder: Model, unet: Model, vae_decoder: Model, state_dict, alpha_list):
        if state_dict != None:
            ov_unet = core.read_model(unet)
            ov_text_encoder = core.read_model(text_encoder)
            ##===Add lora weights===
            visited = []
            lora_dict = {}
            lora_dict_list = []
            LORA_PREFIX_UNET = "lora_unet"
            LORA_PREFIX_TEXT_ENCODER = "lora_te"
            flag = 0
            manager = Manager()
            for iter in range(len(state_dict)):
                visited = []
                for key in state_dict[iter]:
                    if ".alpha" in key or key in visited:
                        continue
                    if "text" in key:
                        layer_infos = key.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split(".")[0]
                        lora_dict = dict(name=layer_infos)
                        lora_dict.update(type="text_encoder")
                    else:
                        layer_infos = key.split(LORA_PREFIX_UNET + "_")[1].split('.')[0]
                        lora_dict = dict(name=layer_infos)
                        lora_dict.update(type="unet")
                    pair_keys = []
                    if "lora_down" in key:
                        pair_keys.append(key.replace("lora_down", "lora_up"))
                        pair_keys.append(key)
                    else:
                        pair_keys.append(key)
                        pair_keys.append(key.replace("lora_up", "lora_down"))

                        # update weight
                    if len(state_dict[iter][pair_keys[0]].shape) == 4:
                        weight_up = state_dict[iter][pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                        weight_down = state_dict[iter][pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                        lora_weights = alpha_list[iter] * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                        lora_dict.update(value=lora_weights)
                    else:
                        weight_up = state_dict[iter][pair_keys[0]].to(torch.float32)
                        weight_down = state_dict[iter][pair_keys[1]].to(torch.float32)
                        lora_weights = alpha_list[iter] * torch.mm(weight_up, weight_down)
                        lora_dict.update(value=lora_weights)
                    #check if this layer has been appended in lora_dict_list
                    for ll in lora_dict_list:
                        if ll["name"] == lora_dict["name"]:
                            ll["value"] += lora_dict["value"] # all lora weights added together
                            flag = 1
                    if flag == 0:
                        lora_dict_list.append(lora_dict)
                    # update visited list
                    for item in pair_keys:
                        visited.append(item)
                    flag = 0
            manager.register_pass(InsertLoRA(lora_dict_list))
            if (True in [('type','text_encoder') in l.items() for l in lora_dict_list]):
                manager.run_passes(ov_text_encoder)
            self.text_encoder = core.compile_model(ov_text_encoder, device)
            manager.run_passes(ov_unet)
            self.unet = core.compile_model(ov_unet, device)
        else:
            self.text_encoder = core.compile_model(text_encoder, device)
            self.unet = core.compile_model(unet, device)

        self.text_encoder_out = self.text_encoder.output(0)
        self.controlnet = core.compile_model(controlnet, device)
        self.unet_out = self.unet.output(0)
        self.vae_decoder = core.compile_model(vae_decoder)
        self.vae_decoder_out = self.vae_decoder.output(0)
    def prepare_image(self, image: Image, batch_size: int, num_images_per_prompt:int = 1, do_classifier_free_guidance: bool = True):
        orig_width, orig_height = image.size
        image, pad = preprocess(image)
        height, width = image.shape[-2:]
        if image.shape[0] == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        image = image.repeat(repeat_by, axis=0)
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))
        return image, height, width, pad, orig_height, orig_width


    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Image.Image,
        num_inference_steps: int = 10,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = [0.0], #single controlnet
        control_guidance_end: Union[float, List[float]] = [1.0], #single controlnet
        eta: float = 0.0,
        latents: Optional[np.array] = None,
        output_type: Optional[str] = "pil",
    ):

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        image, height, width, pad, orig_height, orig_width = self.prepare_image(image, batch_size, do_classifier_free_guidance)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            latents,
        )

         # 6.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0]) #keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.controlnet_pip
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                #text_embeddings = np.split(text_embeddings, 2)[1] if do_classifier_free_guidance else text_embeddings

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
                
                result = self.controlnet([latent_model_input, t, text_embeddings, image, cond_scale])
                down_and_mid_blok_samples = [sample * cond_scale for _, sample in result.items()]

                # predict the noise residual
                noise_pred = self.unet([latent_model_input, t, text_embeddings, *down_and_mid_blok_samples])[self.unet_out]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred,2) #noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents, pad)

        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
        else:
            image = [cv2.resize(img, (orig_width, orig_width))
                     for img in image]

        return image

    def _encode_prompt(self, prompt:Union[str, List[str]], num_images_per_prompt:int = 1, do_classifier_free_guidance:bool = True, negative_prompt:Union[str, List[str]] = None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self.text_encoder_out]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(self, batch_size:int, num_channels_latents:int, height:int, width:int, dtype:np.dtype = np.float32, latents:np.ndarray = None):

        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents:np.array, pad:Tuple[int]):

        latents = 1 / 0.18215 * latents # 1 / self.vae.config.scaling_factor * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image

args = parse_args()
controlnet = ControlNetModel.from_pretrained("sd-controlnet-canny", torch_dtype=torch.float32).cpu()
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5", controlnet=controlnet)

tokenizer = CLIPTokenizer.from_pretrained('clip-vit-large-patch14')
scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

CONTROLNET_OV_PATH = "controlnet-canny.xml"
TEXT_ENCODER_OV_PATH = "text_encoder.xml"
UNET_OV_PATH = "unet_controlnet.xml"
VAE_DECODER_OV_PATH = "vae_decoder.xml"

core = Core()
#core.set_property({'CACHE_DIR': './cache'})
#====Add lora======
LORA_PATH = []
LORA_ALPHA = []
if args.lora_path != "":
    #lora_pair = dict(path=args.lora_path)
    #lora_pair.update(alpha=args.alpha)
    LORA_PATH.append(args.lora_path)
    LORA_ALPHA.append(args.alpha)
    if args.lora_path2 != "":
        #lora_pair = dict(path=args.lora_path2)
        #lora_pair.update(alpha=args.alpha2)
        LORA_PATH.append(args.lora_path2)
        LORA_ALPHA.append(args.alpha)

state_dict = []
# load LoRA weight from .safetensors
if len(LORA_PATH) == 0:
    ov_pipe = OVContrlNetStableDiffusionPipeline(tokenizer, scheduler, core, CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, UNET_OV_PATH, VAE_DECODER_OV_PATH, None, None, device="AUTO") #change to CPU or GPU
else:
    [state_dict.append(load_file(p)) for p in LORA_PATH] #state_dict is list of lora list
    ov_pipe = OVContrlNetStableDiffusionPipeline(tokenizer, scheduler, core, CONTROLNET_OV_PATH, TEXT_ENCODER_OV_PATH, UNET_OV_PATH, VAE_DECODER_OV_PATH, state_dict, LORA_ALPHA, device="AUTO") #change to CPU or GPU

image = load_image("vermeer_512x512.png")
image = np.array(image)

low_threshold = 150
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
#image.save("canny.png")

prompt = ["Girl with Pearl Earring","Girl with blue cloth"]
#prompt = ["Girl with Pearl Earring","1girl, cute, beautiful face, portrait, cloudy mountain, outdoors, trees, rock, river, (soul card:1.2), highly intricate details, realistic light, trending on cgsociety,neon details, ultra realistic details, global illumination, shadows, octane render, 8k, ultra sharp"]
num_steps = 20

negtive_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality","monochrome, lowres, bad anatomy, worst quality, low quality"]
#negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality","3d, cartoon, lowres, bad anatomy, bad hands, text, error"]


np.random.seed(42)
start = time.time()
results = ov_pipe(prompt, image, num_steps, negative_prompt)
end = time.time()-start
print("Inference time({}its): {} s".format(num_steps,end))

for i in range(len(results)):
    results[i].save("result"+str(i)+".png")
