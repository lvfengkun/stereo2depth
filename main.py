import os
import cv2
import numpy as np

left_path = "F:\\BaiduNetdiskDownload\\08\\image_2"  # replace with the path to your left image folder
right_path = "F:\\BaiduNetdiskDownload\\08\\image_3"  # replace with the path to your right image folder
output_path = "depth"  # replace with the path to your output folder

# Get all the left and right images
left_images = [f for f in os.listdir(left_path) if os.path.isfile(os.path.join(left_path, f))]
right_images = [f for f in os.listdir(right_path) if os.path.isfile(os.path.join(right_path, f))]

# Read all the left and right images and convert them to depth maps
for i, (left_file, right_file) in enumerate(zip(left_images, right_images)):
   if left_file.endswith(".jpg") or left_file.endswith(".png") and right_file.endswith(".jpg") or right_file.endswith(
           ".png"):
      if(left_file != right_file):
          print(1)

      left_img = cv2.imread(os.path.join(left_path, left_file))
      right_img = cv2.imread(os.path.join(right_path, right_file))

      stereo = cv2.StereoSGBM_create(minDisparity=0,
                                     numDisparities=128,
                                     blockSize=15,
                                     uniquenessRatio=10,
                                     speckleWindowSize=100,
                                     speckleRange=32,
                                     disp12MaxDiff=1,
                                     P1=8 * 3 * 7 ** 2,
                                     P2=32 * 3 * 7 ** 2
                                     )

      disparity = stereo.compute(left_img, right_img)

      # Normalize the image for representation
      min = disparity.min()
      max = disparity.max()
      disparity = np.uint8(255 * (disparity - min) / (max - min))
      cv2.imshow('disparity', disparity)
      cv2.waitKey(1)

      # Save the depth map to the output folder
      cv2.imwrite(os.path.join(output_path, "{}".format(left_file)), disparity)

#left_imgs_path="F:\BaiduNetdiskDownload\08\image_2"
#right_imgs_path="F:\BaiduNetdiskDownload\08\image_3"