import argparse
import numpy as np
import os
from PIL import Image
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def main(args):
    generated_path = args.generated_images #'/home/luisa/Uni/3dcv project/code/test images'
    real_path = args.real_images

    g_images = [i for i in os.listdir(generated_path) if i.endswith('.png')]
    g_imgs = [i for i in os.listdir(generated_path) if i.endswith('png')]
    g_imgs_np = np.array([np.array(Image.open(os.path.join(generated_path, i))) for i in g_imgs])

    generated_images = torch.tensor(g_imgs_np)
    generated_images = generated_images.permute(0, 3, 1, 2)




    r_images = [i for i in os.listdir(real_path) if i.endswith('.png')]
    r_imgs = [i for i in os.listdir(real_path) if i.endswith('png')]
    r_imgs_np = np.array([np.array(Image.open(os.path.join(real_path, i))) for i in r_imgs])

    real_images = torch.tensor(r_imgs_np)
    real_images = real_images.permute(0, 3, 1, 2)



    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    print(f"FID: {float(fid.compute())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--real_images", type = str, help= "path to dir with real world dataset", required = True)
    parser.add_argument("--generated_images", type = str, help= "path to dir with generated image dataset",  required = True)

    args = parser.parse_args()

    main(args)
