# we take conversation history from IEMOCAP and rewrite its response to be longer and more expressive
import os
import json
import openai
from tqdm import tqdm



emo2text = {
    "neu": "Neutral",
    "hap": "Happy",
    "ang": "Angry",
    "sad": "Sad",
    "exc": "Happy",
    "fru": "Frustrated",
    "sur": "Surprised",
    "fea": "Fearful",
    "dis": "Disgusted",
    "oth": "Other",
    "xxx": "Undecided"
}

used_emotion = set(["Neutral", "Happy", "Sad", "Angry", "Frustrated"])

emotion_label_count = {}

def update_emotion_count(label):
    """Updates the count of an emotion label in the global dictionary."""
    global emotion_label_count
    if label in emotion_label_count:
        emotion_label_count[label] += 1
    else:
        emotion_label_count[label] = 1

def load_json(file_path):
    ret = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            key = data['key'] # e.g. "Ses01F_impro01_F000", used to match with transcript key for re-write
            ret[key] = data
    return ret

def load_transcript(path, info):
    ret = []
    conv_history = []
    counter = 0
    # conv_prefix = 5
    prev_speaker = None
    prev_speaker_emotion = None
    question = None
    num_err = 0
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith("Ses"):
                continue
            try:
                key, duration, text = line.strip().split(' ', 2)
            except:
                print(f"Invalid line in transcript: {line}")
                continue
            counter += 1
            
            if key not in info.keys():
                print(f"Key {key} not found in info dictionary.")
                num_err += 1
                continue

            cur_speaker = "Male" if key.split('_')[-1].startswith("M") else "Female"
            cur_info = info[key]

            emotion = emo2text[cur_info['emo']]
            update_emotion_count(emotion)
            if counter == 1: # skip the first line as no conv history yet
                cur_conv = f"[{cur_speaker} Speaker]: {text}"
                conv_history.append(cur_conv)
                prev_speaker = cur_speaker
                prev_speaker_emotion = emotion
                question = text
                continue # no history for the first line
            
            if cur_speaker == prev_speaker:
                # the conversation continues, we keep track of the conversation but skip the current line
                conv_history.append(f"[{cur_speaker} Speaker]: {text}")
                question = text
                prev_speaker = cur_speaker
                prev_speaker_emotion = emotion
                continue


            if len(conv_history) > 10:
                # we only take the last 10 lines of conv history
                conv_history = conv_history[-10:]

            # we don't want the agent to follow a non-used emotion or very short question
            question_len = len(question.split())
            if prev_speaker_emotion in used_emotion and question_len > 3:
                cur_info['conv_history'] = '\n'.join(conv_history)
                cur_info['gender'] = cur_speaker
                cur_info['original_response'] = text
                cur_info['question'] = question
                cur_info['emo'] = prev_speaker_emotion
                cur_info['original_emo'] = emotion
                ret.append(cur_info) # keep the sequential order
            
            conv_history.append(f"[{cur_speaker} Speaker]: {text}")
            question = text
            prev_speaker = cur_speaker
            prev_speaker_emotion = emotion
            
    return ret, num_err

def generate_audio_response(conversation_entry):
    # Build the prompt that includes conversation history and current speaker's emotion.

    # audio_path = conversation_entry['audio']
    speaker_gender = conversation_entry['gender']
    emotion = conversation_entry['emo']
    orignal_text = conversation_entry['original_response']

    prompt = (
        f"Your task is to continue the following conversation naturally, taking into account both the conversation history and the provided emotional tone. "
        f"You will generate a response from the perspective of a {speaker_gender} speaker, ensuring that your tone reflects a {emotion} emotion. "
        f"An example response is also provided for reference; if it is very short, please expand it to one or two sentences.\n\n"
        f"Conversation History:\n{conversation_entry['conv_history']}\n\n"
        f"Current Speaker's Emotion: {emotion}\n\n"
        f"Example Response: {orignal_text}\n\n"
        f"Now, generate a natural and concise response that continues the dialogue using all of the above information. "
        f"Keep your answer to approximately two sentences. Output the resposne only.\n\n"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",  # Replace with the correct identifier for your GPT‑4-o textual model
        messages=[
            {"role": "system", "content": "You are a helpful conversational assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    message = response['choices'][0]['message']['content']
    return message


def process_script(script, info, output_dir):
    script_id = script.split('/')[-1].split('.')[0]
    out_file = os.path.join(output_dir, f"{script_id}.jsonl")
    if os.path.exists(out_file):
        print(f"File {out_file} already exists. Skipping processing.")
        return 0
    
    script_data, num_err = load_transcript(script, info)
    # now that we have the full script, we can rewrite the response with GPT
    ret = {}
    num_conv = 0
    for conv_data in tqdm(script_data, total=len(script_data)):
        num_conv += 1
        # if num_conv == 10:
        #     break # debug purpose

        try:
            response = generate_audio_response(conv_data)
            key = conv_data['key']
            out_data = conv_data
            out_data['gpt_response'] = response
            ret[key] = out_data
        except Exception as e:
            print(f"Error generating response for {conv_data['key']}: {e}")


    with open(out_file, 'w') as f:
        for key in ret.keys():
            data_to_write = ret[key]
            del data_to_write['conv_history'] # remove the conv history as it is too long
            f.write(json.dumps(data_to_write) + '\n')
    print(f"Processed script {script_id} with {len(ret)} conversations.")
    return num_err





if __name__ == "__main__":
    # we first load the prepared json that has audio/visual feature path along with the labels
    processed_dir = "YOUR_PROCESSED_DIR"
    original_dir = "YOUR_IEMOCAP_DIR"
    all_data_json = "YOUR_DATA_JSON"
    data = load_json(all_data_json)
    output_dir="YOUR_OUTPUT_DIR"

    num_transcripts = 0
    num_err = 0
    for session in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        transcription_dir = os.path.join(original_dir, session, "dialog", "transcriptions")
        transcripts = [file for file in os.listdir(transcription_dir) if file.endswith(".txt") and file.startswith("Ses")]
        num_transcripts += len(transcripts)
        print()
        print("Processing session:", session)
        for script in transcripts:
            # start retriving conversation and redo the response
            script_path = os.path.join(transcription_dir, script)
            process_script(script_path, data, output_dir)

    print(emotion_label_count)
    print(f"Processed {num_transcripts} transcripts")
            