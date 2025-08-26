# I realize that many of the pre-train audio data's ASR are incorrect, we run newest Whisper model to clean them
import os
import sys
sys.path.insert(0, "PATH_TO_AVLM")
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = "PATH_TO_SPIRITLM_CHECKPOINTS"
import torch
from tqdm import tqdm
import json
import numpy as np
import time
import multiprocessing as mp
from pathlib import Path
# from insightface.app import FaceAnalysis
from decord import VideoReader
import cv2
# from insightface.model_zoo import get_model
from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F
from torch import cpu

# Hardcoded paths and settings

output_dir = "YOUR_OUTPUT_DIR"
processed_dir = "YOUR_PROCESSED_DIR"
num_workers = 8 # Adjust based on your GPU count
splits = ["train"]



def get_video_path(visual_path, split):
    key = visual_path.split('/')[-1].replace('.npy', '.mp4')
    folder = visual_path.split('/')[-2]
    seg_pretrain_dir="YOUR_SEG_PRETRAIN_DIR"
    trainval_dir="YOUR_TRAINVAL_DIR"
    test_dir="YOUR_TEST_DIR"
    if split == "train":
        if os.path.exists(os.path.join(seg_pretrain_dir, folder, key)):
            visual_path = os.path.join(seg_pretrain_dir, folder, key)
        elif os.path.exists(os.path.join(trainval_dir, folder, key)):
            visual_path = os.path.join(trainval_dir, folder, key)
        else:
            print(f"Visual path {visual_path} does not exist")
            return None
    elif split == "valid":
        if os.path.exists(os.path.join(trainval_dir, folder, key)):
            visual_path = os.path.join(trainval_dir, folder, key)
        else:
            print(f"Visual path {visual_path} does not exist")
            return None
    elif split == "test":
        if os.path.exists(os.path.join(test_dir, folder, key)):
            visual_path = os.path.join(test_dir, folder, key)
        else:
            print(f"Visual path {visual_path} does not exist")
            return None
    return visual_path


def process_chunk(task):
    chunk_data, split, gpu_id = task
    
    # Initialize model once per process
    device = torch.device(f"cuda:{gpu_id}")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    for data in tqdm(chunk_data, desc=f"Process {gpu_id}", total=len(chunk_data), disable=gpu_id != 0):
        video_path = data['visual_path']

        # video_path = get_video_path(visual_path, split)
        # if video_path is None:
        #     continue
            
        key, filename = video_path.split('/')[-2:]
        output_dir = os.path.join("YOUR_OUTPUT_DIR", split, key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename.replace('.mp4', '.npy'))

        # if os.path.exists(output_file):
        #     # check if we need to reprocess the video
        #     continue
        
        if video_path is not None:
            video_reader = VideoReader(video_path)
            total_frames = len(video_reader)
            if total_frames > 60 * 25:
                continue  # filter out videos that are too long
            
            # Get all frames
            frame_indices = list(range(0, total_frames))
            frames = video_reader.get_batch(frame_indices)
            frames = frames.asnumpy()
            
            # Process frames
            frames = np.array([cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2RGB) for frame in frames])
            batch_tensor = torch.from_numpy(frames).float() / 255.0
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)
            batch_tensor = batch_tensor.to(device)

            with torch.no_grad():
                embeddings = model(batch_tensor)
                np.save(output_file, embeddings.cpu().numpy())


def process_all_files(split, transcript_data):
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    redo_dir = Path("YOUR_REDO_DIR")

    # Split data into chunks for each GPU
    chunk_size = len(transcript_data) // num_gpus
    chunks = [transcript_data[i:i + chunk_size] for i in range(0, len(transcript_data), chunk_size)]
    filtered_chunks = []
    for chunk in chunks:
        cur_chunk = []
        for data in chunk:
            key = data['key']
            file_id = data['file_id'] # 00001_1.wav for example
            if os.path.exists(os.path.join(redo_dir, key, file_id)):
                data['visual_path'] = os.path.join(redo_dir, key, file_id.replace('.wav', '.mp4'))
                cur_chunk.append(data)
        filtered_chunks.append(cur_chunk)

    # Create tasks list
    tasks = []
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id} chunk size: {len(chunks[gpu_id])}")
        tasks.append((filtered_chunks[gpu_id], split, gpu_id))
    
    # process_chunk(tasks[0])
    # Process usng Pool
    with mp.Pool(num_gpus) as pool:
        pool.map(process_chunk, tasks)

def load_jsonl(jsonl_path):
    ret = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ret.append(data)
    return ret


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        print(f"Processing {split} split")
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        existing_transcript = os.path.join(processed_dir, f"{split}.jsonl")
        
        # Load transcript data
        transcript_data = load_jsonl(existing_transcript)
        
        # Process all files for this split
        process_all_files(split, transcript_data)
        
        print(f"Finished {split}")



