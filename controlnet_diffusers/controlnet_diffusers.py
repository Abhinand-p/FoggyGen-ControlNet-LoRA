# !pip install opencv-python transformers accelerate
from diffusers.utils import load_image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from transformers import pipeline
import cv2
from PIL import Image
import os
import random
from diffusers import UniPCMultistepScheduler
import torchvision.transforms as tt
#from accelerate import Accelerator



num_inference_steps = 50

prompt = "street scene with dense fog, high quality photography, Canon EOS R3, street, natural lighting, detailed shadows, natural colors, 35 mm lens, shift"

negative_prompt ="monochrome, unrealistic, bad anatomy, bad quality, low quality, painting, drawing, digital art, 3d render"

# download an image
#image = load_image(
  # "ControlNet-XS_files/shift_car_eg.png"
#)

size = 400

use_gt_depth = True


def get_image(path, size=512):
    image = Image.open(path)
    #if not image.mode == "RGB":
        #image = image.convert("RGB")
    img_size = (size, int(size*1.5))
    image = tt.Resize(img_size)(image)
    image =  ((255.0 - np.array(image)) * 0.8)
    return Image.fromarray(image)
    
# initialize the models and pipeline
# Set the number of images to generate per prompt
num_images_per_prompt = 1

# Set the seed for reproducibility
seed = 202442

# Set the controlnet_conditioning_scale (lambda)
controlnet_conditioning_scale = 0.7

num_samples = 100 #n depth samples to generate from
USE_GT_DEPTH = True
  
def load_depth_map (depthmap_filepath, size = 512):
    
    size = size
    filepath = depthmap_filepath
    
    depth_map = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    B, G, R = cv2.split(depth_map)
    depth_map = 255.0 / ( (256 * 256 * B + 256 * G + R) / (256 * 256 * 256 - 1) * 500 )
    size = (size, int(size * 1.5))
    depth = cv2.resize(depth_map, size , interpolation=cv2.INTER_AREA).astype(np.uint8)
    depth = Image.fromarray(depth_map.astype(np.uint8))
    
    return depth

if USE_GT_DEPTH:
  #load ground truth depth
  depth_path_dir = "train_lora_shift/heavy_fog_final_shift/depth"
  images = []
  paths = []
  lst = os.listdir(depth_path_dir)
  random.shuffle(lst)
  for path in lst:
    #load n depthMaps
    #image = get_image(os.path.join(depth_path_dir,path), size=size)
    #image = load_depth_map("train_lora_shift/00000440_depth_front.png")
    image = load_depth_map(os.path.join(depth_path_dir,path), size=size)
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


#pipe.load_lora_weights("train_lora_shift/sd1.5/finetune/lora/shift/checkpoint-1500", weight_name="pytorch_lora_weights.safetensors", adapter_name="shift")

pipe.to("cuda:1")
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



batch_size = 4

def batch_iterator(dataset, batch_size=1):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


for j,batch in enumerate(batch_iterator(images, batch_size)):
  # Generate images in batches
  generated_images = pipe(
      prompt=[prompt] * len(batch),
      num_inference_steps=num_inference_steps,
      guidance_scale=3.5,
      num_images_per_prompt=num_images_per_prompt,
      negative_prompt=[negative_prompt] * len(batch),
      image=batch,
      generator=torch.manual_seed(seed),
      guess_mode=True,
      controlnet_conditioning_scale=controlnet_conditioning_scale,
  ).images
  
  directory_path = "controlnet_generated_images/vanilla/"
  if not os.path.exists(directory_path):
      os.makedirs(directory_path)
  
  # Save the generated images
  for i, image in enumerate(generated_images):
      image.save(f"{directory_path}/generated_image_{j}_{i}.png")

  print(f"batch progress: {j+1}/{len(images)}")
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
#image_grid(images, num_samples, 1).save("controlnet_lora_samples/orig_depth_maps.png")

img = image_grid(generated_images[:4], 2, 2)
img.save("controlnet_lora_samples/controlnet_vanilla_lora_sd1.5_final.png")