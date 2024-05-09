# !pip install opencv-python transformers accelerate
from diffusers.utils import load_image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import scripts.control_utils as cu
from transformers import pipeline
import cv2
from PIL import Image
import os
import random
from diffusers import UniPCMultistepScheduler


num_inference_steps = 100

prompt = "street scene with dense fog, high quality photography, Canon EOS R3, street, natural lighting, detailed shadows, natural colors, 35 mm lens"

negative_prompt ="monochrome, unrealistic, bad anatomy, bad quality, low quality, painting, drawing, digital art, 3d render"

# download an image
#image = load_image(
  # "ControlNet-XS_files/shift_car_eg.png"
#)

size = 512

use_gt_depth = True



# initialize the models and pipeline
# Set the number of images to generate per prompt
num_images_per_prompt = 5

# Set the seed for reproducibility
seed = 202442

# Set the controlnet_conditioning_scale (lambda)
controlnet_conditioning_scale = 0.7

num_samples = 4 #n depth samples to generate from
USE_GT_DEPTH = True
  
def load_depth_map (depthmap_filepath, size = 512):
    
    size = size
    filepath = depthmap_filepath
    
    depth_map = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    B, G, R = cv2.split(depth_map)
    depth_map = 255.0 / ( (256 * 256 * B + 256 * G + R) / (256 * 256 * 256 - 1) * 500 )
    #depth = cv2.resize(depth_map, (size,size), interpolation=cv2.INTER_AREA).astype(np.uint8)
    depth = Image.fromarray(depth_map.astype(np.uint8))
    
    return depth

if USE_GT_DEPTH:
  #load ground truth depth
  depth_path_dir = "train_lora_shift/clear_shift_1000/depth"
  images = []
  paths = []
  lst = os.listdir(depth_path_dir)
  random.shuffle(lst)
  for path in lst:
    #load n depthMaps
    #image = cu.get_image(os.path.join(depth_path_dir,path), size=size)
    image = load_depth_map("train_lora_shift/00000440_depth_front.png")
    images.append(image)
    paths.append(path)
    if len(images) >= num_samples:
      break
  print("depth maps loaded...")
  with open(f"controlnet_samples/depth_map_list.txt",'w') as fp:
    fp.write("\n".join(paths))
    
else:
  #extract using DE models
  
  img_path_dir = "train_lora_shift/heavy_fog_shift_1200/images"
  images = []
  for path in os.listdir(img_path_dir):
    #load n depthMaps
    image = cu.get_image(os.path.join(img_path_dir,path), size=size)
    depth_estimator = pipeline('depth-estimation')
  
    #image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    images.append(image)
    if len(images) >= num_samples:
      break
  
  


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


pipe.load_lora_weights("train_lora_shift/sd1.5/finetune/lora/shift", weight_name="pytorch_lora_weights.safetensors", adapter_name="shift")

pipe.to("cuda")
pipe.enable_model_cpu_offload()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
    

prompts = [prompt] * num_samples
negative_prompts = [negative_prompt] * num_samples

assert len(prompts) == len(images)
# Generate images in batches
generated_images = pipe(
    prompt=prompts,
    num_inference_steps=num_inference_steps,
    guidance_scale=3.5,
    num_images_per_prompt=num_images_per_prompt,
    negative_prompt=negative_prompts,
    image=images,
    generator=torch.manual_seed(seed),
    guess_mode=True,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
).images

# Save the generated images
for i, image in enumerate(generated_images):
    image.save(f"controlnet_samples/generated_image_{i}.png")

#resize images to original sizes
#generated_images = []  
#original_size = (1200, 800)
#for img in generated_images:
#  resized_image = img.resize(original_size, Image.LANCZOS)
#  generated_images.append(resized_image)

#depth_imgs = []
#for img in images:
#  resized_image = img.resize(original_size, Image.LANCZOS)
#  depth_imgs.append(resized_image)

#save depth maps
image_grid(images, num_samples, 1).save("controlnet_samples/orig_depth_maps.png")

img = image_grid(generated_images, num_samples, num_images_per_prompt)
img.save("controlnet_samples/controlnet_vanilla_lora_1.5.png")