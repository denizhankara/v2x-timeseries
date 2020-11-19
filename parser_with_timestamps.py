import json
import argparse
import os
import numpy as np
from parse import parse
from glob import glob
from typing import List, Dict
from progressbar import ProgressBar
import pandas as pd
import tsfresh


PATH_TO_SCEN = "work/ul/ul_vertsys/ul_wqy57/" # path to scenario root
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

def vehicleParser(json_file,path_to_json):


    log_name_tokens = os.path.splitext(json_file)[0].split("-")
    vehicle_id = log_name_tokens[1] # [0]:= 'JSONlog' [1]:=vehicle instance count [2]:=vehicle id [1]:=vehicle attack type
    vehicle_OMNET_ID = log_name_tokens[2] # [0]:= 'JSONlog' [1]:=vehicle instance count [2]:=vehicle id [1]:=vehicle attack type
    
    attack_type = log_name_tokens[3]
    if attack_type == 'A0':
        is_vehicle_attacker = 0
    else:
        is_vehicle_attacker= 1 

    # start parsing the file
    vehicleList= []
    file_path = path_to_json+json_file
    with open(file_path, "r") as f:
        log_rows = f.readlines()

    for row in log_rows:    
        json_row = json.loads(row)
        json_row['vehicle_ID'] = vehicle_id # set here, change later
        json_row['vehicle_OMNET_ID'] = vehicle_OMNET_ID
        json_row['is_attacker'] = is_vehicle_attacker
        if json_row['type'] == 2 :
            pass
        else:
            
            # extract position 
            json_row['pos_x'] = json_row['pos'][0] 
            json_row['pos_y'] = json_row['pos'][1] 
            json_row['pos_z'] = json_row['pos'][2] 
            del json_row['pos']
            # extract speed 
            json_row['spd_x'] = json_row['spd'][0] 
            json_row['spd_y'] = json_row['spd'][1] 
            json_row['spd_z'] = json_row['spd'][2] 
            del json_row['spd']
            
            # extract position noise
            json_row['pos_x_noise'] = json_row['pos_noise'][0] 
            json_row['pos_y_noise'] = json_row['pos_noise'][1] 
            json_row['pos_z_noise'] = json_row['pos_noise'][2] 
            del json_row['pos_noise']

            # extract spd noise 
            json_row['spd_x_noise'] = json_row['spd_noise'][0] 
            json_row['spd_y_noise'] = json_row['spd_noise'][1] 
            json_row['spd_z_noise'] = json_row['spd_noise'][2] 
            del json_row['spd_noise']

            # add new dict to list
            vehicleList.append(json_row)
            
            
    current_df = pd.DataFrame(vehicleList)
            
    # returns a dataframe

    return current_df
    

def parseAndExport(jsondir,exportfilename): # currently manual name, will be automated eventually

    big_df = pd.DataFrame()
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and pos_json.startswith('JSONlog'))]
    for json_file in json_files :
        parsed_df = vehicleParser(json_file,path_to_json) #.sort_values(["sender","rcvTime"])
        if not parsed_df.empty:
            parsed_df = parsed_df.sort_values(["sender","rcvTime"])
        big_df = pd.concat([big_df,parsed_df], axis=0, ignore_index=True)

    
    big_df.to_csv(index=False,path_or_buf=exportfilename)

    # Done parsing
    print("Done")
    return big_df
    

if __name__ == "__main__":
    
    print("Start parsing as time series")
    # get the files list 
    # big_df = pd.DataFrame()
    #path_to_JSON_folder= "work/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14506023.180205_200518/veins-maat/simulations/securecomm2018/results/JSONlog-0-7-A0.json"
    
    #exit()
    # get all the JSON files from directory
    
    #Automate later 
    path_to_json = 'work/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505247.180205_165727/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json,"veins_maat.uc1.14505247.180205_165727.csv")
    
    path_to_json = 'work 2/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505342.180205_171010/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json,"veins_maat.uc1.14505342.180205_171010.csv")
    
    path_to_json = 'work 3/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505511.180205_173553/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json,"veins_maat.uc1.14505511.180205_173553.csv")
    
    path_to_json = 'work 4/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505930.180205_192941/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json,"veins_maat.uc1.14505930.180205_192941.csv")
    
    path_to_json = 'work 5/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14506016.180205_200240/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json,"veins_maat.uc1.14506016.180205_200240.csv")
    
    print("All done")




    
    """
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and pos_json.startswith('JSONlog'))]
    for json_file in json_files :
        parsed_df = vehicleParser(json_file,path_to_json) #.sort_values(["sender","rcvTime"])
        if not parsed_df.empty:
            parsed_df = parsed_df.sort_values(["sender","rcvTime"])
        big_df = pd.concat([big_df,parsed_df], axis=0, ignore_index=True)

    #big_df.to_csv(index=False,path_or_buf='export.csv')
    
    # Do feature extraction and selection

    #big_df = big_df.sort_values(["sender","rcvTime","vehicle_ID"])
    big_df.to_csv(index=False,path_or_buf='export.csv')

    # Done parsing
    print("Done")
    """