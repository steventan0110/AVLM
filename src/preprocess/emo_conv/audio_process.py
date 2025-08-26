import os
import json
import shutil
import sys
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set SPIRITLM checkpoints directory from environment or default
if 'SPIRITLM_CHECKPOINTS_DIR' not in os.environ:
    os.environ['SPIRITLM_CHECKPOINTS_DIR'] = os.environ.get('SPIRITLM_CHECKPOINTS_DIR', './checkpoints/spiritlm')
from src.exp.spiritlm.spiritlm.speech_tokenizer import spiritlm_expressive
import torchaudio.transforms as T
import multiprocessing


def load_audio(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize waveform
    waveform = waveform / torch.max(torch.abs(waveform))
    return waveform


def process_audio(file, tokenizer):
    audio = load_audio(file)
    try:
        units, style_str, base_str = tokenizer.encode_string(audio)
    except Exception as e:
        logging.error(f"Error processing audio {file}: {e}")
        return None
    return style_str


def initialize_gpu_worker(gpu_id):
    """Initialize a worker process with a GPU-specific tokenizer"""
    device = f"cuda:{gpu_id}"
    logging.info(f"Initializing tokenizer on {device}")
    
    # Initialize tokenizer on the specific GPU and store it globally for this process
    global tokenizer
    tokenizer = spiritlm_expressive(device=device)
    
    # Set the device for this process
    torch.cuda.set_device(gpu_id)


def process_file_on_gpu(file_path, prev_dict, gpu_id, output_file):
    """Process a single file on a specific GPU"""
    # Process is initialized with a GPU-specific tokenizer
    data = load_json(file_path)
    logging.info(f"Process on GPU {gpu_id}: Processing {len(data)} items from {os.path.basename(file_path)}")
    
    video_dir = os.environ.get('VIDEO_DIR', './data/videos')
    audio_dir = os.environ.get('AUDIO_DIR', './data/audio')
    rewrite_dir = os.environ.get('REWRITE_DIR', './data/rewrite')
    
    # To avoid file locking issues, write to a temporary file first
    temp_output = f"{output_file}_{gpu_id}_{os.path.basename(file_path)}.tmp"
    
    with open(temp_output, "w") as f:
        processed_count = 0
        for item in data:
            key = item['key']
            if key not in prev_dict:
                continue
            
            question_key = prev_dict[key]
            question_audio_path = os.path.join(audio_dir, f"{question_key}.wav")
            respone_audio_path = os.path.join(rewrite_dir, f"{key}.wav")
            question_video_path = os.path.join(video_dir, f"{question_key}.avi")
            
            if not os.path.exists(question_audio_path) or not os.path.exists(respone_audio_path) or not os.path.exists(question_video_path):
                logging.warning(f"Skipping {key} because of missing audio or video files")
                continue
                
            item['question_audio_path'] = question_audio_path
            item['response_audio_path'] = respone_audio_path
            item['question_video_path'] = question_video_path

            # Process the audio data into units using the process-specific tokenizer
            question_units = process_audio(question_audio_path, tokenizer)
            response_units = process_audio(respone_audio_path, tokenizer)
            if question_units is None or response_units is None:
                logging.warning(f"Skipping {key} because of None units")
                continue
            
            item['question_units'] = question_units
            item['response_units'] = response_units
            f.write(json.dumps(item) + "\n")
            processed_count += 1
    
    return temp_output, processed_count


def load_json(file):
    ret = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            ret.append(data)
    return ret


def load_prev_dict():
    ret = {}
    original_dir = os.environ.get('IEMOCAP_DIR', './data/IEMOCAP')
    for session in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        transcription_dir = os.path.join(original_dir, session, "dialog", "transcriptions")
        transcripts = [file for file in os.listdir(transcription_dir) if file.endswith(".txt") and file.startswith("Ses")]
   
        for script in transcripts:
            # start retriving conversation and redo the response
            script_path = os.path.join(transcription_dir, script)
            # print(script_path)
            prev_key = None
            with open(script_path, "r") as f:
                for line in f:
                    if not line.startswith("Ses"):
                        continue
                    else:
                        key = line.strip().split(' ', 2)[0]
                        if prev_key is not None:
                            ret[key] = prev_key
                        prev_key = key
    return ret


def launcher_function(gpu_id, files, prev_dict, output_file, temp_files_list):
    """Function to launch the GPU worker process"""
    _process_files_on_gpu(files, prev_dict, gpu_id, output_file, temp_files_list)


if __name__ == "__main__":
    # Set number of GPUs for processing
    num_gpus = 4
    
    # Verify GPU availability
    if torch.cuda.device_count() < num_gpus:
        logging.warning(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
        num_gpus = torch.cuda.device_count()
    
    logging.info(f"Using {num_gpus} GPUs for processing")
    
    prev_dict = load_prev_dict()
    data_path = os.environ.get('DATA_PATH', './data/input')
    output_dir = os.environ.get('OUTPUT_DIR', './data/processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "data.jsonl")
    
    # Get all JSONL files to process
    json_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jsonl")]
    
    # Split files among GPUs
    files_per_gpu = {}
    for i, file in enumerate(json_files):
        gpu_id = i % num_gpus
        if gpu_id not in files_per_gpu:
            files_per_gpu[gpu_id] = []
        files_per_gpu[gpu_id].append(file)
    
    # Create a context that allows starting processes with a specific initializer
    mp_context = multiprocessing.get_context('spawn')  # 'spawn' is required for CUDA
    
    # Create a manager to handle the shared list
    manager = mp_context.Manager()
    temp_files = manager.list()
    
    # Create processes, one per GPU, each handling multiple files
    processes = []
    
    for gpu_id in range(num_gpus):
        if gpu_id not in files_per_gpu:
            continue
            
        processor = mp_context.Process(
            target=launcher_function,
            args=(gpu_id, files_per_gpu[gpu_id], prev_dict, output_file, temp_files)
        )
        processes.append(processor)
    
    # Start and join processes
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
    
    # Combine all temporary files into the final output
    with open(output_file, "w") as final_file:
        total_processed = 0
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    for line in f:
                        final_file.write(line)
                        total_processed += 1
                os.remove(temp_file)
    
    logging.info(f"Processed {total_processed} items in total")


def _process_files_on_gpu(files, prev_dict, gpu_id, output_file, temp_files_list):
    """Process multiple files on a specific GPU with a single tokenizer instance"""
    # Initialize the GPU and tokenizer for this process
    device = f"cuda:{gpu_id}"
    print(f"Process {os.getpid()} initializing tokenizer on {device}")
    
    # Initialize tokenizer on the specific GPU
    global tokenizer
    tokenizer = spiritlm_expressive(device=device)
    
    # Set the device for this process
    torch.cuda.set_device(gpu_id)
    
    # Process each file
    local_temp_files = []
    total_processed = 0
    
    for file_path in files:
        print(f"Process on GPU {gpu_id}: Processing file {os.path.basename(file_path)}")
        
        # Process file and get temp output file path
        temp_output = f"{output_file}_{gpu_id}_{os.path.basename(file_path)}.tmp"
        processed_count = _process_single_file(file_path, prev_dict, temp_output)
        
        local_temp_files.append(temp_output)
        total_processed += processed_count
    
    # Update the shared list of temp files
    temp_files_list.extend(local_temp_files)
    
    print(f"Process on GPU {gpu_id} finished processing {total_processed} items across {len(files)} files")
    
    # Clean up
    del tokenizer
    torch.cuda.empty_cache()


def _process_single_file(file_path, prev_dict, temp_output):
    """Process a single file using the already initialized tokenizer"""
    data = load_json(file_path)
    print(f"Processing {len(data)} items from {os.path.basename(file_path)}")
    
    video_dir = os.environ.get('VIDEO_DIR', './data/videos')
    audio_dir = os.environ.get('AUDIO_DIR', './data/audio')
    rewrite_dir = os.environ.get('REWRITE_DIR', './data/rewrite')
    
    processed_count = 0
    with open(temp_output, "w") as f:
        for item in data:
            key = item['key']
            if key not in prev_dict:
                continue
            
            question_key = prev_dict[key]
            question_audio_path = os.path.join(audio_dir, f"{question_key}.wav")
            respone_audio_path = os.path.join(rewrite_dir, f"{key}.wav")
            question_video_path = os.path.join(video_dir, f"{question_key}.avi")
            
            if not os.path.exists(question_audio_path) or not os.path.exists(respone_audio_path) or not os.path.exists(question_video_path):
                logging.warning(f"Skipping {key} because of missing audio or video files")
                continue
                
            item['question_audio_path'] = question_audio_path
            item['response_audio_path'] = respone_audio_path
            item['question_video_path'] = question_video_path

            # Process the audio data into units using the process-specific tokenizer
            question_units = process_audio(question_audio_path, tokenizer)
            response_units = process_audio(respone_audio_path, tokenizer)
            if question_units is None or response_units is None:
                logging.warning(f"Skipping {key} because of None units")
                continue
            
            item['question_units'] = question_units
            item['response_units'] = response_units
            f.write(json.dumps(item) + "\n")
            processed_count += 1
    
    return processed_count
