import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
def load_jsonl(jsonl_path):
    ret = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ret[data['key']] = data
    return ret


if __name__ == "__main__":
    cur_data_dir = "YOUR_DATA_DIR"
    val_data = f"{cur_data_dir}/valid.jsonl"
    test_data = f"{cur_data_dir}/test.jsonl"
    train_data = f"{cur_data_dir}/large_train.jsonl"
    output_dir = f"{cur_data_dir}/filter_large"

    val_manifest = load_jsonl(val_data)
    test_manifest = load_jsonl(test_data)
    train_manifest = load_jsonl(train_data)
    manifest = train_manifest
    num_filtered = 0
    write_handle = open(f"{output_dir}/train.jsonl", "w")
    for key, data in tqdm(manifest.items(), total=len(manifest)):
        video_path = data['video_path']
        smirk_path = data['smirk_path']
        features = np.load(smirk_path, allow_pickle=True).item()
        pose_param = features['pose_params'] # T, 3
        # jaw_params = features['jaw_params'] # T, 3

        needs_filter = False
        all_yaws = []
        for i in range(pose_param.shape[0]):
            pose_param_i = pose_param[i]
            rotation = R.from_rotvec(pose_param_i)
            # yaw: left/right;  pitch: up/down;  roll: clockwise/counterclockwise
            yaw, pitch, roll = rotation.as_euler('yxz', degrees=True)
            all_yaws.append(yaw)
        all_yaws = np.array(all_yaws)
        mean_yaw = np.mean(all_yaws)
        if mean_yaw > 30:
            num_filtered += 1
            continue
        else:
            write_handle.write(json.dumps(data) + "\n")
    write_handle.close()
    print(f"num_filtered: {num_filtered}")
    
      


