import os
import json
import shutil
import sys
from tqdm import tqdm
sys.path.insert(0, "PATH_TO_AVLM")
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = "PATH_TO_SPIRITLM_CHECKPOINTS"
from src.exp.spiritlm.spiritlm.speech_tokenizer import spiritlm_expressive
from src.exp.spiritlm.spiritlm.speech_tokenizer.spiritlm_tokenizer 

def process_audio(tokenizer, file):
    # process the audio file with spirtlm tokenizer
    units = tokenizer.encode_units(file)
    dedup_str = tokenizer.encode_string(file)
    if len(dedup_str) == 0:
        return None
    str_len = len(dedup_str.split('[')) - 1
    ret = units
    ret['unit_len'] = str_len
    ret['unit_str'] = dedup_str
    return ret


if __name__ == "__main__":
    tokenizer = spiritlm_expressive()
    audio_dir = "YOUR_AUDIO_DIR"
    # for split in ['test', 'valid', 'train']:
    for split in ['train']:
        cur_dir = os.path.join(audio_dir, split)
        json_handle = open(os.path.join(cur_dir, "speech_units.json"), 'w')
        subdirs = [d for d in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir, d))]
        for subdir in tqdm(subdirs, total=len(subdirs)):
            for file in os.listdir(os.path.join(cur_dir, subdir)):
                audio_file = os.path.join(cur_dir, subdir, file)
                try:
                    ret = process_audio(tokenizer, audio_file)
                except:
                    print('Error processing file: ', audio_file)
                    continue
                if ret is None:
                    print('Error processing file: ', audio_file)
                    continue
                ret['key'] = subdir
                ret['file_id'] = file
                json_handle.write(json.dumps(ret) + '\n')
        json_handle.close()
    process_audio(tokenizer, file)