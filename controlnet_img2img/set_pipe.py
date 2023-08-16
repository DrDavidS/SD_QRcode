from PIL import Image
from os.path import exists
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler
from diffusers.utils import load_image


class qrPipeline():
    def __init__(self, init_img_path:str, controlnet_id:str="DionTimmer/controlnet_qrcode-control_v11p_sd21", stable_diffusion_id: str="stabilityai/stable-diffusion-2-1") -> None:
        self.controlnet_id = controlnet_id
        self.stable_diffusion_id = stable_diffusion_id
        self.init_img_path = init_img_path
    
    def pretrained_pipeline(self):
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=self.stable_diffusion_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")
        pipe.enable_model_cpu_offload()
        return pipe
    
    def cut_squar_img(self)->Image:
        if not exists(self.init_img_path):
            raise FileNotFoundError("The init_image dose not exist. Check path or file name again.")
        im = Image.open(self.init_img_path)

        short_edge = min(im.size)
        x1, y1, x2, y2 = 0, 0, short_edge, short_edge
        im_croped = im.crop((x1, y1, x2, y2))
        return im_croped
    
    def resize_for_condition_image(self, resolution: int=768)->Image:
        input_image = self.cut_squar_img()
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img 
    



