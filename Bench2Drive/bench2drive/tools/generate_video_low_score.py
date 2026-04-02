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

def get_low_score_entries(merged_json_path, score_threshold=50):
    """Read merged.json and return entries with score_composed < score_threshold"""
    with open(merged_json_path, 'r') as f:
        data = json.load(f)
    
    low_score_entries = []
    for record in data['_checkpoint']['records']:
        if record['scores']['score_composed'] < score_threshold:
            # The save_name in JSON doesn't have the bench2drive220_X_ prefix
            # We need to find the corresponding folder name
            save_name = record['save_name']
            low_score_entries.append(save_name)
    
    return low_score_entries

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dir', required=True, help='path to evaluation output directory')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--score_threshold', type=float, default=50)
    args = parser.parse_args()
    fps = args.fps
    folder_dir = args.folder_dir
    merged_json_path = os.path.join(folder_dir, 'merged.json')
    
    # Create output folder for low-score videos
    output_folder = os.path.join(folder_dir, 'low_score_videos')
    os.makedirs(output_folder, exist_ok=True)
    
    low_score_entries = get_low_score_entries(merged_json_path, score_threshold=args.score_threshold)
    print(f"Found {len(low_score_entries)} entries with score < {args.score_threshold}")
    
    # Generate videos only for low-score entries
    for folder_path in glob.glob(folder_dir + '/*'):
        if os.path.isdir(folder_path):
            folder_name = os.path.basename(folder_path)
            # Check if this folder corresponds to a low-score entry
            # The folder name contains the save_name as a substring
            is_low_score = any(save_name in folder_name for save_name in low_score_entries)
            
            if is_low_score:
                output_path = os.path.join(output_folder, f"{folder_name}.mp4")
                print(f"Generating video for low-score entry: {folder_name}")
                create_video(folder_path, output_path, fps)
            else:
                print(f"Skipping {folder_name} (score >= 50)")
    
    print(f"\nVideos saved to: {output_folder}")
