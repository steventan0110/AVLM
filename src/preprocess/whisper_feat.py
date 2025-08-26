import argparse
import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
from transformers import WhisperProcessor, WhisperModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for CosyVoice")
    parser.add_argument("--dataset", type=str, choices=["iemocap", "recola", "multidialog"], default="iemocap")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input audio files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output audio files",
    )
    return parser.parse_args()


def prepare_data(input_dir):
    data_dict = defaultdict(list)
    for split in ['train', 'test']:
        jsonl_path = f"{input_dir}/{split}/vit.jsonl"
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                data_dict[split].append(data)
    return data_dict


def extract_whisper_features(wav_path, whisper_processor, whisper_model, device, chunk_duration=30.0, overlap=0.0):
    """
    Extracts speech features using Whisper Large v3 encoder, handling both short and long audios.
    Splits long audios into overlapping 30s chunks and extracts valid features.
    """
    # Load audio

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

    # Compute total duration
    audio_duration = waveform.shape[1] / 16000  # Whisper uses 16kHz

    # Define chunking parameters
    chunk_samples = int(chunk_duration * 16000)  # 30s * 16kHz
    overlap_samples = int(overlap * 16000)  # 5s * 16kHz

    # Split waveform into chunks
    start = 0
    chunk_features = []

    while start < waveform.shape[1]:
        end = min(start + chunk_samples, waveform.shape[1])
        chunk = waveform[:, start:end]

        # Extract features for this chunk
        input_features = whisper_processor(chunk.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features.to(device)
        # Extract encoder outputs
        with torch.no_grad():
            encoder_outputs = whisper_model.encoder(input_features).last_hidden_state  # Shape: (1, 3000, D)

        # Compute valid range
        chunk_duration_sec = (end - start) / 16000
        valid_mel_frames = int((chunk_duration_sec / 30.0) * 1500)
        valid_mel_frames = max(1, min(valid_mel_frames, 3000))

        # Store valid features
        valid_encoder_outputs = encoder_outputs[:, :valid_mel_frames, :].squeeze(0).cpu().numpy()
        chunk_features.append(valid_encoder_outputs)

        # Move start position
        start += chunk_samples - overlap_samples  # Shift with overlap

    # Concatenate all chunks
    full_features = np.concatenate(chunk_features, axis=0)  # Shape: (T_total, D)
    fps = full_features.shape[0] / audio_duration
    return full_features


def process_batch(data_batch, whisper_processor, whisper_model, feat_dir, device):
    """Process batch of audio files using Whisper encoder"""
    wav_paths = [data["audio"] for data in data_batch]
    keys = [data["key"] for data in data_batch]

    # Extract Whisper embeddings
    speech_features = [extract_whisper_features(wav_path, whisper_processor, whisper_model, device) for wav_path in wav_paths]
    
    processed_data = []
    for i, key in enumerate(keys):
        output_path = os.path.join(feat_dir, f"{key}.npy")

        np.save(output_path, speech_features[i])
        data_to_write = data_batch[i]
        data_to_write['speech_feat'] = output_path
        data_to_write['whisper_len'] = speech_features[i].shape[0]
        processed_data.append(data_to_write)    
    return processed_data


@torch.no_grad()
def process_iemocap(args):
    data_dict = prepare_data(args.input_dir)

    # Load Whisper Large v3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device)

    batch_size = 16  # Adjust as needed
    
    for split in ["train", "test"]:
        logging.info(f"Currently processing split: {split}")
        output_dir = os.path.join(args.output_dir, split)
        os.makedirs(output_dir, exist_ok=True)
        feat_dir = os.path.join(output_dir, "features")
        os.makedirs(feat_dir, exist_ok=True)
        write_handle = open(os.path.join(output_dir, "data.jsonl"), "w")
        
        data_list = data_dict[split]
        for i in tqdm(range(0, len(data_list), batch_size)):
            batch = data_list[i:i+batch_size]
            processed_data = process_batch(batch, whisper_processor, whisper_model, feat_dir, device)
            for item in processed_data:
                write_handle.write(json.dumps(item) + '\n')

        write_handle.close()


@torch.no_grad()
def process_recola(args):
    data_jsonl = os.path.join(args.input_dir, "data.jsonl")
    with open(data_jsonl, "r") as f:
        data_list = [json.loads(line) for line in f]
    logging.info(f"Total number of audio files: {len(data_list)}")
    train_dir, dev_dir = os.path.join(args.output_dir, "train"), os.path.join(args.output_dir, "dev")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(dev_dir, "features"), exist_ok=True)

    # Load Whisper Large v3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device)

    train_handle = open(os.path.join(args.output_dir, "train", "data.jsonl"), "w")
    dev_handle = open(os.path.join(args.output_dir, "dev", "data.jsonl"), "w")
    for item in data_list:
        audio_path = item['audio']
        key = audio_path.split("/")[-1].split(".")[0] # dev_6_35_36
        if "train" in key:
            split = "train"
        else:
            split = "dev"
        audio_feat = extract_whisper_features(audio_path, whisper_processor, whisper_model, device)
        output_path = os.path.join(args.output_dir, split, "features", f"{key}.npy")
        np.save(output_path, audio_feat)
        item['speech_feat'] = output_path
        item['whisper_len'] = audio_feat.shape[0]
        if split == "train":
            train_handle.write(json.dumps(item) + '\n')
        else:
            dev_handle.write(json.dumps(item) + '\n')
    train_handle.close()
    dev_handle.close()
    logging.info("Processing completed!")



if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "iemocap":
        process_iemocap(args)
    elif args.dataset == "recola":
        process_recola(args)
