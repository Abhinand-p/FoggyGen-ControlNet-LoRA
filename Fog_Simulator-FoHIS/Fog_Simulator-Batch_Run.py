import numpy as np
import math
import cv2
from noise import pnoise3
from PIL import Image
import os

# Visibility range of molecules in meters
molecule_visibility = 12
# Visibility range of aerosols in meters
aerosol_visibility = 450
# Coefficients
ECM = 3.912 / molecule_visibility
ECA = 3.912 / aerosol_visibility
# FOG_TOP m
FT = 70
# HAZE_TOP m
HT = 34
# These parameters are set based for road-scenes based images
CAMERA_ALTITUDE = 1.6
fov = 64


# Function to read image DPI
def get_image_info(src):
    im = Image.open(src)
    try:
        dpi = im.info['dpi']
    except KeyError:
        dpi = (72, 72)
    return dpi


# Function to estimate elevation, distance, and angle
def elevation_and_distance_estimation(src, depth, vertical_fov, camera_altitude):
    # Load the image and get its DPI
    img = cv2.imread(src)
    img_dpi = get_image_info(src)

    height, width = img.shape[:2]
    altitude = np.empty((height, width))
    distance = np.empty((height, width))
    angle = np.empty((height, width))
    depth_min = depth.min()

    for j in range(width):
        for i in range(height):
            # theta is the vertical angle
            theta = i / (height - 1) * vertical_fov
            # Case: theta is less than half of the vertical FOV
            if theta < 0.5 * vertical_fov:
                distance[i, j] = depth[i, j] / math.cos(math.radians(0.5 * vertical_fov - theta))
                h_half = math.tan(0.5 * vertical_fov) * depth_min
                y2 = (0.5 * height - i) / img_dpi[0] * 2.56
                y1 = h_half * y2 / (height / img_dpi[0] * 2.56)
                altitude[i, j] = camera_altitude + depth[i, j] * y1 / depth_min
                angle[i, j] = 0.5 * vertical_fov - theta
            # Case: theta is half of the vertical FOV
            elif theta == 0.5 * vertical_fov:
                distance[i, j] = depth[i, j]
                h_half = math.tan(0.5 * vertical_fov) * depth_min
                y2 = (i - 0.5 * height) / img_dpi[0] * 2.56
                y1 = h_half * y2 / (height / img_dpi[0] * 2.56)
                altitude[i, j] = max(camera_altitude - depth[i, j] * y1 / depth_min, 0)
                angle[i, j] = 0
            # Case: theta is greater than half of the vertical FOV
            elif theta > 0.5 * vertical_fov:
                distance[i, j] = depth[i, j] / math.cos(math.radians(theta - 0.5 * vertical_fov))
                h_half = math.tan(0.5 * vertical_fov) * depth_min
                y2 = (i - 0.5 * height) / img_dpi[0] * 2.56
                y1 = h_half * y2 / (height / img_dpi[0] * 2.56)
                altitude[i, j] = max(camera_altitude - depth[i, j] * y1 / depth_min, 0)
                angle[i, j] = -(theta - 0.5 * vertical_fov)
    return altitude, distance, angle


# Function to generate Perlin noise
def noise(Ip, depth):
    p1 = Image.new('L', (Ip.shape[1], Ip.shape[0]))
    p2 = Image.new('L', (Ip.shape[1], Ip.shape[0]))
    p3 = Image.new('L', (Ip.shape[1], Ip.shape[0]))
    scales = [1 / 130.0, 1 / 60.0, 1 / 10.0]
    for i, scale in enumerate(scales):
        for y in range(Ip.shape[0]):
            for x in range(Ip.shape[1]):
                v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
                color = int((v + (1.2 if i == 2 else (0.5 if i == 1 else 1))) * 128)
                if i == 0:
                    p1.putpixel((x, y), color)
                elif i == 1:
                    p2.putpixel((x, y), color)
                elif i == 2:
                    p3.putpixel((x, y), color)
    perlin = (np.array(p1) + np.array(p2) / 2 + np.array(p3) / 4) / 3
    return perlin


# Directory paths for images and depth maps
image_dir = 'Results/SHIFT_FINAL_FOG/images'
depth_map_dir = 'Results/SHIFT_FINAL_FOG/depth'

# Iterate over each image file
for image_file in os.listdir(image_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png') or image_file.endswith('.jpeg'):
        common_part_image = '_'.join(image_file.split('_')[
                                     1:-1])

        depth_map_file = next((filename for filename in os.listdir(depth_map_dir) if
                               filename.endswith('.png') and f"depth_{common_part_image}" in filename), None)

        if depth_map_file:
            # Construct full paths
            image_path = os.path.join(image_dir, image_file)
            depth_map_path = os.path.join(depth_map_dir, depth_map_file)

            print("-Image File:", image_path)
            print("-Depth File:", depth_map_path)

            # Load image and depth map
            Ip = cv2.imread(image_path)
            Dp = cv2.imread(depth_map_path)

            # Preprocess depth map
            depth = Dp[:, :, 0].astype(np.float64)
            depth[depth == 0] = 1
            depth *= 3

            # Calculate elevation, distance, and angle
            elevation, distance, angle = elevation_and_distance_estimation(image_path, depth, fov, CAMERA_ALTITUDE)

            # Generate Perlin noise
            perlin = noise(Ip, depth)

            # Calculate fog parameters
            c = (1 - elevation / (FT + 0.00001))
            c[c < 0] = 0
            ECM = (ECM * c + (1 - c) * ECA) * (perlin / 255)

            # Calculate distances through fog and haze
            distance_through_fog = np.zeros_like(distance)
            distance_through_haze = np.zeros_like(distance)
            distance_through_haze_free = np.zeros_like(distance)

            idx1 = (np.logical_and(FT > elevation, elevation > HT))
            idx2 = elevation <= HT
            idx3 = elevation >= FT

            distance_through_haze[idx2] = distance[idx2]
            distance_through_fog[idx1] = (elevation[idx1] - HT) * distance[idx1] / (elevation[idx1] - CAMERA_ALTITUDE)
            distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
            distance_through_haze[idx3] = (HT - CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_fog[idx3] = (FT - HT) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

            # Apply fog effect to image in RGB color space
            I = np.empty_like(Ip)
            result = np.empty_like(Ip)

            I[:, :, 0] = Ip[:, :, 2] * np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)  # Red channel
            I[:, :, 1] = Ip[:, :, 1] * np.exp(
                -ECA * distance_through_haze - ECM * distance_through_fog)  # Green channel
            I[:, :, 2] = Ip[:, :, 0] * np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)  # Blue channel

            O = 1 - np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)

            Ial = np.empty_like(Ip)
            Ial[:, :, 0] = 201  # R
            Ial[:, :, 1] = 225  # G
            Ial[:, :, 2] = 225  # B

            result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]  # R
            result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]  # G
            result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]  # B

            print(f'--Writing Results/Simulated_FINAL_FOG/{image_file}_result.jpg')
            cv2.imwrite(f'Results/Simulated_FINAL_FOG/{image_file}_result.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            print(f"No corresponding depth map found for {image_file}")
