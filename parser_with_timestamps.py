import json
import argparse
import os
import numpy as np
from parse import parse
from glob import glob
from typing import List, Dict
from progressbar import ProgressBar

PATH_TO_SCEN = "/ul/ul_vertsys/ul_wqy57/" # path to scenario root
PATH_TO_JSON = "/veins-maat/simulations/securecomm2018/results/"
SCEN_PREFIX = "veins_maat.uc1." # the prefix of the scenario
INDEX_TRIGGER = "| --- | --- | --- | --- | --- |"
ROW_FORMAT = "| {} | {} | {} | {} | [{}]({}) |"

def hash_attr(rep:int, den_level:int, att_type:int, att_den:float)->str:
    return "rep:%d,den_level:%d,att_type:%d,att_den:%f"%(rep,den_level, att_type, att_den)

# extract features as specified in "ML Based Approach to Detect Position Falsification Attack in VANETs"
# By Singh et al.
# Ground truth is not needed
def feature_extraction_singh(json_file:str, extraction_type: int, malicious_vehicles: Dict):
    output = []
    # Read the Log file
    with open(json_file, "r") as f:
        log_rows = f.readlines()
    current_state = None
    for row in log_rows:
        json_row = json.loads(row)
        if json_row['type'] == 2:
            # local generated state
            current_state = json_row
            continue

        label = 0
        if json_row['sender'] in malicious_vehicles:
            label = 1

        if extraction_type == 1:
            # get position and speed
            pos_spd = np.array(json_row['pos']+json_row['spd'])
            r = np.concatendate[np.append(np.array([label]), pos_spd)].tolist()
            output.append(r)
            continue
        elif extraction_type == 2:
            # get posiiton and position diff from the sender
            pos = np.array(json_row['pos'])
            current_pos = np.array(current_state['pos'])
            delta_pos = pos - current_pos
            r = np.concatenate([np.array([label]), pos,delta_pos]).tolist()
            output.append(r)
            continue
        elif extraction_type ==3:
            # get position, speed, pos diff, and speed diff
            pos_spd = np.array(json_row['pos']+json_row['spd'])
            current_pos_spd = np.array(current_state['pos'] + current_state['spd'])
            delta_pos_spd = pos_spd - current_pos_spd
            rssi = json_row['RSSI']
            r = np.concatenate([np.array([label]), pos_spd, delta_pos_spd, np.array([np.array(rssi)])]).tolist()
            output.append(r)
            continue
        else:
            exit("Undefined singh extraction type: %d"%(extraction_type))
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script parses VeReMi dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='The root directory of the dataset. The directory should be named \"work/\"')
    parser.add_argument('--index', type=str, required=True, help="The path to index.md as specified in the VeReMi github dataset")
    parser.add_argument('--feature_extraction', type=str, default='singh', help="The path to index.md as specified in the VeReMi github dataset")
    parser.add_argument('--output', type=str, default='./parsed/', help="The directory of the parsed output saved as .npy")
    parser.add_argument('--overwrite', type=int, default=0, help="0 := Do not overwrite preprocessed data, 1 := Overwite preprocessed data")
    parser.add_argument('--save_as_numpy', type=bool, default=False, help="Flag to save the parsed data as numpy")
    ARGS = parser.parse_args()
    
    # Check if the dataset directory is correct
    dataset_scen_root =  ARGS.dataset_root + PATH_TO_SCEN
    print(dataset_scen_root)
    if not os.path.basename(ARGS.dataset_root) == "work" or not os.path.isdir(dataset_scen_root):
        exit("The specified path %s is not the correct dataset root"%(ARGS.dataset_root))

    # Check if the index file is correct
    index_content = []
    index_trigger_flag = False
    if not os.path.isfile(ARGS.index):
        exit("The specified index file does not exist")
    if not os.path.isdir(ARGS.output):
        os.mkdir(ARGS.output)

    with open(ARGS.index, "r") as f:
        for l in f.readlines():
            if not index_trigger_flag and INDEX_TRIGGER not in l:
                continue
            elif index_trigger_flag:
                index_content.append(l.strip())
            elif INDEX_TRIGGER in l:
                index_trigger_flag = True
    # Build 2 dicts
    ATTR_TO_DIRNAME = {}
    DIRNAME_TO_ATTR = {}
    for row in index_content:
        # Check format
        rep, density_level, attacker_type, attacker_density, dirname, _ = parse(ROW_FORMAT, row)
        rep = int(rep)
        density_level = int(density_level)
        attacker_type = int(attacker_type)
        attacker_density = float(attacker_density)
        dirname = parse("{}.tgz", dirname)[0]
        ATTR_TO_DIRNAME[hash_attr(rep,density_level, attacker_type,attacker_density)] = dirname
        DIRNAME_TO_ATTR[dirname] = {"rep":rep,"density_level":density_level,"attacker_type":attacker_type,"attacker_density":attacker_density}
    with open(ARGS.output + "index.json",'w') as f:
        json.dump(DIRNAME_TO_ATTR,f,indent=4)

    # Get Scenario Names and parse the data content
    DIRNAME_TO_DATA={}
    total_scenarios = len(glob(dataset_scen_root+SCEN_PREFIX+"*"))
    pbar = ProgressBar()
    scen_count = 0
    for scen_full_path in pbar(glob(dataset_scen_root+SCEN_PREFIX+"*")):
        scen_count += 1
        scen_name = os.path.basename(scen_full_path)
        if scen_name not in DIRNAME_TO_ATTR:
            exit("Unexpected directory encountered: %s"%(scen_full_path))
        if not os.path.isdir(scen_full_path+PATH_TO_JSON):
            exit("The directory %s does not have path to json data: expected dir %s"%(scen_full_path, scen_full_path+PATH_TO_JSON))

        # Check if the file already exists. If it does, check if the overwite is set
        parsed_scen_output_dir = os.path.join(ARGS.output, scen_name + ".json")
        if os.path.isfile(parsed_scen_output_dir) and ARGS.overwrite == 0:            
            continue

        # Check for json files
        ground_truth_dir = scen_full_path+PATH_TO_JSON+"GroundTruthJSONlog.json"
        if not os.path.isfile(ground_truth_dir):
            exit("Expected GroundTruthJSONlog.json at dir %s but does not exist"%(scen_full_path+PATH_TO_JSON))

        vehicles_logs = {}
        attacker_id = set()
        ground_truth = []
        ground_truth_index_by_id = {}
        with open(ground_truth_dir, "r") as f:
            ground_truth_rows = f.readlines()
        for row in ground_truth_rows:
            msg = json.loads(row)
            if msg['attackerType'] != 0:
                attacker_id.add(msg['sender'])
            ground_truth.append(msg)
            ground_truth_index_by_id[msg["messageID"]] = msg
        assert len(ground_truth) == len(ground_truth_index_by_id)

        # Get files names of each vehicle logs
        for d in glob(scen_full_path+PATH_TO_JSON+"*.json"):
            if d == ground_truth_dir:
                continue
            log_name_tokens = os.path.splitext(os.path.basename(d))[0].split("-")
            JSONlog = log_name_tokens[1] # [0]:= 'JSONlog' [1]:=vehicle instance count [2]:=vehicle id [1]:=vehicle attack type
            vehicle_id = log_name_tokens[2] # [0]:= 'JSONlog' [1]:=vehicle instance count [2]:=vehicle id [1]:=vehicle attack type
            vlog = feature_extraction_singh(d, 3, attacker_id)
            vehicles_logs[vehicle_id]= vlog
        vehicles_logs['attr'] = DIRNAME_TO_ATTR[scen_name]
        with open(parsed_scen_output_dir, 'w') as f:
            if not ARGS.save_as_numpy:
                json.dump(vehicles_logs, f) 
            else:
                np.save(f, vehicles_logs)
    print(scen_count)
    print(len(DIRNAME_TO_ATTR))
    assert scen_count == len(DIRNAME_TO_ATTR.keys())
