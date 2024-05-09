"""SHIFT dataset generation finetune image with text prompt"""

"""Download shift dataset using this:
 python download.py --view "[front]" --group "[img, det_2d, depth]" --split "[val]" --framerate "[images]" --shift "discrete"
"""

import os
import sys

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import shutil
import cv2

from tqdm.auto import tqdm

# Add the root directory of the project to the path. Remove the following two lines
# if you have installed shift_dev as a package.
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend


def main():
    """Load the SHIFT dataset and print the tensor shape of the first batch."""

    dataset = SHIFTDataset(
        data_root="hd_st312-carla_sim/SHIFT_data/",
        split="val",
        keys_to_load=[
            Keys.images,                # note: images, shape (1, 3, H, W), uint8 (RGB)
            #Keys.intrinsics,            # note: camera intrinsics, shape (3, 3)
            Keys.boxes2d,               # note: 2D boxes in image coordinate, (x1, y1, x2, y2)
            Keys.boxes2d_classes,       # note: class indices, shape (num_boxes,)
            Keys.boxes2d_track_ids,     # note: object ids, shape (num_ins,)
            #Keys.boxes3d,               # note: 3D boxes in camera coordinate, (x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z)
            #Keys.boxes3d_classes,       # note: class indices, shape (num_boxes,), the same as 'boxes2d_classes'
            #Keys.boxes3d_track_ids,     # note: object ids, shape (num_ins,), the same as 'boxes2d_track_ids'
            Keys.segmentation_masks,    # note: semantic masks, shape (1, H, W), long
            #Keys.masks,                 # note: instance masks, shape (num_ins, H, W), binary
            Keys.depth_maps,            # note: depth maps, shape (1, H, W), float (meters)
        ],
        views_to_load=["front"],
        shift_type="discrete",          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
        backend=ZipBackend(),           # also supports HDF5Backend(), FileBackend()
        verbose=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )


    seq_df = pd.read_csv("seq.csv")
    
    use_coarse_filter = False #coarse is basically broader weather so it can include more samples in case of fog (both small/heavy) or clear (can be slightly cloud/clear etc)
    
   
    # Filter 'video' column based on 'start_weather' column for only heavy fog
    filter_tag = "heavy fog" #this is based on weather_fine
    filter_condition = "start_weather_coarse"  if use_coarse_filter else "start_weather_fine"
    filtered_videos = seq_df.loc[seq_df[filter_condition] == filter_tag, 'video'].tolist()
   
    
    num_samples = 200  #number of samples to generate for our dataset
    #create dataframe dict
    data_dict = {"image_name":[], "videoName":[], "bb_objects":[], "text":[]}
    
    #save images and depth_maps to a separate folder
    save_to_folder = True
    save_depth = True #whether we want to save depth or speed up by skipping depth saving (eg for lora training) 
    
    #save folder for the new dataset
    save_dir = f"heavy_fog_final_shift"
    
    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    print('\n')
    
    if save_to_folder:
      data_dict["image_path"] = []
      #save rgb vid and Depth map
      # Check if save_dir exists, create it if not
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)
      
      # Check if images and depth folders exist, create them if not
      images_dir = os.path.join(save_dir, 'images')
      depth_dir = os.path.join(save_dir, 'depth')
      
      if not os.path.exists(images_dir):
          os.makedirs(images_dir)
      
    #the information for bounding boxes can be extracted using below code but they have limited classes, more useful information with 23 classes is in semantic segmentation
    k = "boxes2d_classes"
    CLASSES = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")
    
    for i, batch in tqdm(enumerate(dataloader)):
        
        #print(batch["depth"].keys())
        
        #filter for only heavy fog videos and nonzero objects in that frame
        if batch["front"]["videoName"][0] in filtered_videos:
          data_dict["videoName"].append(batch['front']['videoName'][0])
          data_dict["image_name"].append(batch['front']['name'][0])
          
          data_dict['image_path'].append(f"{images_dir}/image_{data_dict['videoName'][-1]}_{data_dict['image_name'][-1]}")
          #data_dict["segmentation_masks"].append(batch['front'][''][0])
          
          
          if batch["front"]["boxes2d_classes"].shape[1] != 0:
            data = batch['front'][k]
            #print(data.shape)
            all_objs = " ".join(set([CLASSES[key] for key in data.numpy().squeeze(0)]))
            
            data_dict["bb_objects"].append(all_objs)
            data_dict["text"].append(f"synthentic street road photo having {filter_tag} with all the objects containing in image such as {all_objs}")
          else:
            data_dict["bb_objects"].append("NA")
            data_dict["text"].append(f"synthentic street road photo having {filter_tag} with streetscape")
            
          #print(f"Batch {i}:\n")
          #print(f"{'Item':20} {'Shape':35} {'Min':10} {'Max':10}")
          #print("-" * 80)
          #print(f"{k:20} {all_objs:35} {data.min():10.2f} {data.max():10.2f}")
          
          
          
          #for k, data in batch["front"].items():
              #if isinstance(data, torch.Tensor):
                  #print(f"{k:20} {str(data.shape):35} {data.min():10.2f} {data.max():10.2f}")
              #else:
               #   print(f"{k:20} {data}")
          
        
          #save rgb vid and Depth map
          if save_to_folder:
          
            #img = batch['front']['images']
            #depth = batch['front']['depth_maps']
            
            #img = img.numpy().squeeze()  # remove batch dimension and convert to numpy
            #img = img.transpose((1, 2, 0))  # transpose to (H, W, 3) format
            #print(data_dict['videoName'], data_dict['image_name'])
            new_path = f"{images_dir}/image_{data_dict['videoName'][-1]}_{data_dict['image_name'][-1]}"
            old_path = f"hd_st312-carla_sim/SHIFT_data/images/{data_dict['videoName'][-1]}/{data_dict['image_name'][-1]}"
            shutil.copy(old_path,new_path)
            #cv2.imwrite(f"{images_dir}/image_{data_dict['videoName'][-1]}_{data_dict['image_name'][-1]}", img)  # save to disk
            
            
            if save_depth:
              if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
              #depth = depth.numpy().squeeze()  # remove batch dimension
              #depth = (depth - depth.min()) / (depth.max() - depth.min())  # normalize to [0, 1]
              #depth_conv = depth.astype(np.float32)  # convert to another format float32
              new_path = f"{depth_dir}/depth_{data_dict['videoName'][-1]}_{data_dict['image_name'][-1][:-4]}.png"
              old_path = f"hd_st312-carla_sim/SHIFT_data/depth/{data_dict['videoName'][-1]}/{data_dict['image_name'][-1][:8]}_depth_front.png"
              shutil.copy(old_path,new_path)
              #cv2.imwrite(f"{depth_dir}/depth_{data_dict['videoName'][-1]}_{data_dict['image_name'][-1][:-4]}.png", depth_conv)  # save depth map as PNG
            
        
        #create only 1k samples
        if i % 500 == 0:
          print(len(data_dict["videoName"]))
        if len(data_dict["videoName"]) >= num_samples : #1k for now but this should be commented to truly be sampling at later stage, 
        #might give error when u try sampling more than 1k later so keep it higher but not too high to iterate very slow
          break
          #else:
            #print(f"skipped {batch['front']['videoName'][0]}")

    # Print the sample indices within a video.
    # The video indices groups frames based on their video sequences. They are useful for training on videos.
    
    
    dataset_name = f"{save_dir}/shift_foggy_images_.2k.csv"
    final_df = pd.DataFrame(data_dict)
    
    
    #selects n no. of rows from final_df
    #final_df = final_df.sample(num_samples)  #sample only n rows and save it later
            
    final_df.to_csv(dataset_name,index=False)
    print(f"dataset saved as {dataset_name}")
    
    
    
    print('\n')
    video_to_indices = dataset.video_to_indices
    for video, indices in video_to_indices.items():
        print(f"Video name: {video}")
        print(f"Sample indices within a video: {indices}")
        break


if __name__ == "__main__":
    main()
