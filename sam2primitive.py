import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载图像文件
image_path = image_path = './images/sds_cats/image1.png'
image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict("cat")