# the EmoBOX train/test jsonl does not cover all keys, we redo the chunk
import os
import json
import re
import cv2
import shutil
from tqdm import tqdm




def extract_frames(input_video, output_video, start_time, end_time, is_female, dominant_gender):
    """
    Extracts the head region from the given portion of the video (male or female) and resizes it to 128x128.
    
    Parameters:
        input_video (str): Path to the input video.
        output_video (str): Path to save the processed video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    
    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if dimensions match expectation (720x480)
    if width != 720 or height != 480:
        print("Unexpected video dimensions. Expected 720x480.")
        cap.release()
        return None

    # Calculate start and end frame
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    # print(f"FPS: {fps}, Total Frames: {total_frames}, Width: {width}, Height: {height}, Start Frame: {start_frame}, End Frame: {end_frame}")

    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        print("Invalid time range.")
        cap.release()
        return None

    # Define region of interest (ROI) based on the cropped frame
    if dominant_gender == "M":
        x_offset = 360 if is_female else 0
    else:
        x_offset = 0 if is_female else 360

    # crop_y_start, crop_y_end = 112, 368  # New height: 256
    crop_y_start, crop_y_end = 112, 368
    roi_width, roi_height = 360, 280  # New cropped size before further trimming
    resolution = 256
    # Load Haar cascade for face detection
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (resolution, resolution))


    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[crop_y_start:crop_y_end, :]
        roi = cropped_frame[:, x_offset:x_offset + roi_width]  

        cropped_roi = roi[:, 52:308] # Now 256x256
        output_frame = cv2.resize(cropped_roi, (resolution, resolution))
        out.write(output_frame)

    cap.release()
    out.release()
  
            

def parse_iemocap_line(line):
    """
    Parses an IEMOCAP transcript line and extracts:
    - Header (including F000/M000, e.g., Ses01F_impro01_F000 or Ses01M_impro06_M000)
    - Start time (converted to seconds)
    - End time (converted to seconds)
    - Text
    
    Args:
        line (str): A transcript line in the format:
                    "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me."
    
    Returns:
        tuple: (header, start_time_seconds, end_time_seconds, text)
    """
    pattern = re.match(r"^(Ses[^\[]+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:\s*(.+)", line, re.DOTALL)

    if pattern:
        header = pattern.group(1).strip()  # Everything before "["
        start_time = float(pattern.group(2))  # Convert to seconds
        end_time = float(pattern.group(3))    # Convert to seconds
        text = pattern.group(4).strip().replace("\n", " ")  # Handle multi-line text
        return header, start_time, end_time, text
    else:
        print("Line format is incorrect:", line)
        return None, None, None, None  # Return None if the line format is incorrect
    

def process_emo(file_path):
    """
    Processes a text file containing conversation data and returns a dictionary
    mapping file identifiers to emotions.

    Expected header line format:
    [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

    For example:
    [6.2901 - 8.2357]    Ses01F_impro01_F000    neu    [2.5000, 2.5000, 2.5000]

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        dict: Dictionary where keys are file identifiers (e.g., 'Ses01F_impro01_F000')
              and values are the corresponding emotions (e.g., 'neu', 'fru', 'xxx').
    """
    # Regular expression to capture the header fields:
    # - The time range is inside brackets.
    # - Followed by the identifier (non-whitespace).
    # - Followed by the emotion (non-whitespace).
    # - And finally the [V, A, D] values (which we ignore).
    pattern = re.compile(
        r'^\s*\[\s*(?P<start_time>[^\]]+)\s*-\s*(?P<end_time>[^\]]+)\s*\]\s+'
        r'(?P<identifier>\S+)\s+'
        r'(?P<emotion>\S+)\s+'
        r'\[.*\]'
    )
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or header lines that start with %
            if not line or line.startswith('%'):
                continue

            match = pattern.match(line)
            if match:
                identifier = match.group('identifier')
                emotion = match.group('emotion')
                result[identifier] = emotion
    return result


def process_script(script_path, emotion_path, video_path, wav_dir, output_dir, write_handle):
    emo_dict = process_emo(emotion_path)
    # load the script
    with open(script_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                header, start_time, end_time, text = parse_iemocap_line(line)
                if header is not None:
                    key = header
                    start = start_time
                    end = end_time
                    if key not in emo_dict:
                        print(f"Key {key} not found in emo_dict.")
                        continue
                    emo_label = emo_dict[key]
                    
                    # process the video and audio 
                    video_outpupt_path = os.path.join(output_dir, "video", f"{key}.avi")
                    dominant_gender = "M" if "M" in key.split('_')[0] else "F"                    
                    is_female = key.split('_')[-1].startswith('F')

   
                    extract_frames(video_path, video_outpupt_path, start, end, is_female, dominant_gender)
                    wav_path = os.path.join(wav_dir, f"{key}.wav")
                    if not os.path.exists(wav_path):
                        print(f"{wav_path} does not exist")
                        continue
                    audio_path = os.path.join(output_dir, "audio", f"{key}.wav")
                    shutil.copy(wav_path, audio_path)
                    data_to_save = {
                        "key": key,
                        "emo": emo_label,
                        "text": text,
                        "audio_path": audio_path,
                        "video_path": video_path,
                    }
                    write_handle.write(json.dumps(data_to_save) + '\n')



if __name__ == "__main__":
    # we first load the prepared json that has audio/visual feature path along with the labels

    original_dir = "YOUR_IEMOCAP_DIR"
    output_dir="YOUR_OUTPUT_DIR" # where we keep the chunked audio and videos
    num_transcripts = 0

    data_json = f"{output_dir}/data.jsonl"
    write_handle = open(data_json, 'w')
    for session in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        transcription_dir = os.path.join(original_dir, session, "dialog", "transcriptions")
        emotion_dir = os.path.join(original_dir, session, "dialog", "EmoEvaluation")
        transcripts = [file for file in os.listdir(transcription_dir) if file.endswith(".txt") and file.startswith("Ses")]
        num_transcripts += len(transcripts)
        print("Processing session:", session)
        for script in tqdm(transcripts, total=len(transcripts)):
            # start retriving conversation and redo the response
            script_path = os.path.join(transcription_dir, script)
            emotion_path = os.path.join(emotion_dir, script)
            assert os.path.exists(script_path), f"{script_path} does not exist"
            assert os.path.exists(emotion_path), f"{emotion_path} does not exist"
            header = script.split('.')[0]
            video_path = os.path.join(original_dir, session,  "dialog/avi/DivX", f"{header}.avi")
            # cap/Session2/sentences/wav/Ses02F_impro07/Ses02F_impro07_F001.wav
            wav_dir = os.path.join(original_dir, session, "sentences/wav", header)
            process_script(script_path, emotion_path, video_path, wav_dir, output_dir, write_handle)
      
    write_handle.close()
    print(f"Processed {num_transcripts} transcripts.")
