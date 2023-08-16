class qrPipeline():
    def __init__(self, controlnet_id:str="DionTimmer/controlnet_qrcode-control_v11p_sd21", stable_diffusion_id: str="stabilityai/stable-diffusion-2-1" ) -> None:
        self.controlnet_id = controlnet_id
        self.stable_diffusion_id = stable_diffusion_id
    
    def pretrained_pipeline(self):
        print(self.controlnet_id)
        print(self.stable_diffusion_id)


qr = qrPipeline()
qr.pretrained_pipeline()