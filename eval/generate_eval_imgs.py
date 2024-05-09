import argparse
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import os
import torch
from diffusers.utils import load_image
import torchvision.transforms as tt

from PIL import Image
import numpy as np
import cv2

NEGATIVE_PROMPT="monochrome, lowres, unrealistic, worst quality, low quality, cartoon, painting, drawing, digital art, black and white"
PROMPT = "streetscape, dense fog, high quality photography, Canon EOS R3, street, natural lighting, detailed shadows, 35 mm lens"

def load_depth_map (depthmap_filepath, size = 512, shift_depth_maps = False):
    depth =  load_image(depthmap_filepath)
    size = size
    filepath = depthmap_filepath
    

      
    depth_map = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    B, G, R = cv2.split(depth_map)
    depth_map = 255.0 / ( (256 * 256 * B + 256 * G + R) / (256 * 256 * 256 - 1) * 500 )

    depth = cv2.resize(depth_map, (size,size), interpolation=cv2.INTER_AREA).astype(np.uint8)
    depth = Image.fromarray(depth)

       
    return depth


def main(args):
    depth_maps_dir = args.depth_maps_dir
    lora_path = args.lora_path
    prompt = PROMPT
    negative_prompt = NEGATIVE_PROMPT
    shift_depth_maps = False

    if args.pos_prompt:
        prompt = args.pos_prompt
    if args.neg_prompt:
        negative_prompt = args.neg_prompt
    if args.shift_depth_maps:
        shift_depth_maps = True


    out_dir = os.path.join('data', os.path.basename(depth_maps_dir) + '_' + os.path.basename(lora_path) )
    os.mkdir(out_dir)

    controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.load_lora_weights(lora_path)
    pipe.to("cuda")

    for i in os.listdir(depth_maps_dir):
        
        depth_fpath = os.path.join(depth_maps_dir, i)
        #print(depth_fpath)

        depth = load_depth_map(depth_fpath, shift_depth_maps =  shift_depth_maps)
        output_image = pipe(prompt, depth, negative_prompt=negative_prompt, num_inference_steps=50).images[0]

        s_path = os.path.join(out_dir, i)
        output_image.save(s_path)



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--depth_maps_dir", type = str, help= "path to dir with depth maps for image generation", required = True)
    parser.add_argument("--lora_path", type = str, help= "path to lora weights", required = True)
    
    parser.add_argument("--pos_prompt", type = str, help= "positive prompt", required = False)
    parser.add_argument("--neg_prompt", type = str, help= "negative prompt", required = False)
    parser.add_argument("--shift_depth_maps", type = bool, help= "True if shift dataset depthmaps are used", required = False)


    args = parser.parse_args()

    main(args)