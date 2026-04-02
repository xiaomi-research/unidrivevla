import os
import cv2
import glob
import json
import numpy as np

from tqdm import trange

def create_video(images_folder, output_video, fps):
    images = [img for img in os.listdir(os.path.join(images_folder, 'images')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    if len(images) > 0:
        frame = cv2.imread(os.path.join(os.path.join(images_folder, 'images'), images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for i in trange(1, len(images)):
            image = images[i]
            img = cv2.imread(os.path.join(os.path.join(images_folder, 'images'), image))

            video.write(img)
        video.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dir', required=True, help='path to evaluation output directory')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    fps = args.fps
    folder_dir = args.folder_dir

    for folder_path in glob.glob(folder_dir + '/*'):
        if os.path.isdir(folder_path):
            output_path = os.path.join(folder_dir, "{}.mp4".format(os.path.basename(folder_path)))
            create_video(folder_path, output_path, fps)
