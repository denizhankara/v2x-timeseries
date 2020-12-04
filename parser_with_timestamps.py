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
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters,EfficientFCParameters
from numpy import loadtxt,savetxt
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,precision_score,recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split,cross_validate,StratifiedKFold


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

def parseAll():
    # Automate later
    path_to_json = 'work/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505247.180205_165727/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json, "veins_maat.uc1.14505247.180205_165727.csv")

    path_to_json = 'work 2/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505342.180205_171010/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json, "veins_maat.uc1.14505342.180205_171010.csv")

    path_to_json = 'work 3/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505511.180205_173553/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json, "veins_maat.uc1.14505511.180205_173553.csv")

    path_to_json = 'work 4/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14505930.180205_192941/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json, "veins_maat.uc1.14505930.180205_192941.csv")

    path_to_json = 'work 5/ul/ul_vertsys/ul_wqy57/veins_maat.uc1.14506016.180205_200240/veins-maat/simulations/securecomm2018/results/'
    parseAndExport(path_to_json, "veins_maat.uc1.14506016.180205_200240.csv")

    print("All done")

    pass

def get_attackers(df):

    attacker_dict = {}
    # get the attacker signals that are sent by attacker vehicles
    for index,row in df.iterrows():
        if row['is_attacker'] == 1:
            attacker_dict[int(row['vehicle_ID'])] = 1
        else:
            attacker_dict[int(row['vehicle_ID'])] = 0


    return attacker_dict


def extractFeatures(df,attacker_filename,full_set=False):
    
    my_set_of_params = {'autocorrelation': [{'lag': 0}, {'lag': 1}, {'lag': 2}, {'lag': 3}, {'lag': 4}, {'lag': 5}, {'lag': 6}, {'lag': 7}, {'lag': 8}, {'lag': 9}],
                        'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'var', 'maxlag': 40}],
                        'number_cwt_peaks' : [{'n': 1}, {'n': 5}],
                        'number_peaks' : [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 50}],
                        'cwt_coefficients':[{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2},
                                             {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5},
                                             {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10},
                                             {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 20},
                                             {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2},
                                             {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 5},
                                             {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 10},
                                             {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 20},
                                             {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 2},
                                             {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5},
                                             {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 10},
                                             {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 20},
                                             {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 2},
                                             {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 5},
                                             {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 10},
                                             {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 20}
                        ],'fft_coefficient':[{'coeff': 0, 'attr': 'real'},
                                             {'coeff': 1, 'attr': 'real'},
                                             {'coeff': 2, 'attr': 'real'},
                                             {'coeff': 3, 'attr': 'real'},
                                             {'coeff': 4, 'attr': 'real'},
                                             {'coeff': 5, 'attr': 'real'},
                                             {'coeff': 6, 'attr': 'real'},
                                             {'coeff': 7, 'attr': 'real'},
                                             {'coeff': 8, 'attr': 'real'},
                                             {'coeff': 9, 'attr': 'real'},
                                             {'coeff': 10, 'attr': 'real'},
                                             {'coeff': 11, 'attr': 'real'},
                                             {'coeff': 12, 'attr': 'real'},
                                             {'coeff': 13, 'attr': 'real'},
                                             {'coeff': 14, 'attr': 'real'},
                                             {'coeff': 15, 'attr': 'real'},
                                             {'coeff': 0, 'attr': 'imag'},
                                             {'coeff': 1, 'attr': 'imag'},
                                             {'coeff': 2, 'attr': 'imag'},
                                             {'coeff': 3, 'attr': 'imag'},
                                             {'coeff': 4, 'attr': 'imag'},
                                             {'coeff': 5, 'attr': 'imag'},
                                             {'coeff': 6, 'attr': 'imag'},
                                             {'coeff': 7, 'attr': 'imag'},
                                             {'coeff': 8, 'attr': 'imag'},
                                             {'coeff': 9, 'attr': 'imag'},
                                             {'coeff': 10, 'attr': 'imag'},
                                             {'coeff': 11, 'attr': 'imag'},
                                             {'coeff': 12, 'attr': 'imag'},
                                             {'coeff': 13, 'attr': 'imag'},
                                             {'coeff': 14, 'attr': 'imag'},
                                             {'coeff': 15, 'attr': 'imag'},
                                             {'coeff': 0, 'attr': 'abs'},
                                             {'coeff': 1, 'attr': 'abs'},
                                             {'coeff': 2, 'attr': 'abs'},
                                             {'coeff': 3, 'attr': 'abs'},
                                             {'coeff': 4, 'attr': 'abs'},
                                             {'coeff': 5, 'attr': 'abs'},
                                             {'coeff': 6, 'attr': 'abs'},
                                             {'coeff': 7, 'attr': 'abs'},
                                             {'coeff': 8, 'attr': 'abs'},
                                             {'coeff': 9, 'attr': 'abs'},
                                             {'coeff': 10, 'attr': 'abs'},
                                             {'coeff': 11, 'attr': 'abs'},
                                             {'coeff': 12, 'attr': 'abs'},
                                             {'coeff': 13, 'attr': 'abs'},
                                             {'coeff': 14, 'attr': 'abs'},
                                             {'coeff': 15, 'attr': 'abs'},
                                             {'coeff': 0, 'attr': 'angle'},
                                             {'coeff': 1, 'attr': 'angle'},
                                             {'coeff': 2, 'attr': 'angle'},
                                             {'coeff': 3, 'attr': 'angle'},
                                             {'coeff': 4, 'attr': 'angle'},
                                             {'coeff': 5, 'attr': 'angle'},
                                             {'coeff': 6, 'attr': 'angle'},
                                             {'coeff': 7, 'attr': 'angle'},
                                             {'coeff': 8, 'attr': 'angle'},
                                             {'coeff': 9, 'attr': 'angle'},
                                             {'coeff': 10, 'attr': 'angle'},
                                             {'coeff': 11, 'attr': 'angle'},
                                             {'coeff': 12, 'attr': 'angle'},
                                             {'coeff': 13, 'attr': 'angle'},
                                             {'coeff': 14, 'attr': 'angle'},
                                             {'coeff': 15, 'attr': 'angle'}

                                             ]

                        }

    # create ID by signal transmission, essentially we have to understand if a signal is from malicious entity
    # Create unique ID column
    df['transmission_ID'] = df['sender'].astype(str) + "_" + df['vehicle_ID'].astype(str)


    """can save attacker dicts and use later"""
    with open(attacker_filename) as f:
        attacker_dict = json.load(f)

    #attacker_dict = get_attackers(df)
    # trim unneccessary columns
    df = df.drop(['sendTime', 'type', 'vehicle_OMNET_ID','is_attacker','messageID','sender','vehicle_ID'], axis=1)
    df["rcvTime"] = df["rcvTime"].astype(int)

    new_df = df # .head(4813)

    # stack the dataframe for supporting different sample freqs

    if full_set:
        extracted_features = extract_features(new_df, column_id="transmission_ID", column_sort="rcvTime")
    else:
        settings = MinimalFCParameters()
        default_fc_parameters = settings.update(my_set_of_params)
        extracted_features = extract_features(new_df, column_id="transmission_ID", column_sort="rcvTime",default_fc_parameters=settings)
        #labels_df = pd.DataFrame(index = extracted_features.index.copy())
    labels = []

    for index,row in extracted_features.iterrows():
        current_sender = index.split('_')[0]
        #new_row = {'malicious':attacker_dict.get(current_sender,0)}
        labels.append(attacker_dict.get(current_sender,0))
#        labels = labels.append(new_row,ignore_index=True)
        #labels.iloc[index] =attacker_dict.get(current_sender,0)
        pass
    labels = np.asarray(labels,dtype=int)
    #labels_df['malicious'] = labels
    impute(extracted_features)
    features_filtered = select_features(extracted_features, labels)
    features_filtered['labels'] = labels
    return features_filtered,labels

def featureExtractorFromcsv(filename,attacker_filename,full_set = False):
    print("Start extracting features from  time series")
    df = pd.read_csv(filename)
    features_filtered, labels = extractFeatures(df,attacker_filename,full_set=full_set)
    print("done feature extraction")
    simulation_case = os.path.splitext(filename)[0]
    feature_file_name = simulation_case +'_features.csv'
    label_file_name = simulation_case + '_labels.csv'
    features_filtered.to_csv(feature_file_name)
    savetxt(label_file_name, labels, delimiter=',')
    print("Done")

    return feature_file_name,label_file_name


def makePrediction(features_file,labels_file):
    features = pd.read_csv(features_file)
    labels = loadtxt(labels_file, delimiter=',')
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
    features = features.drop(['id', 'labels'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        stratify=labels,
                                                        test_size=0.2,random_state=0)
    # scores = cross_val_score(xgb_model, features, labels, scoring=("precision","recall"), cv=10)

    #xgb_model.fit(features, labels)
    xgb_model.fit(X_train,y_train)


    #y_pred = xgb_model.predict(features)
    y_pred = xgb_model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.3f' % accuracy)

    precision = precision_score(y_test, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    recall = recall_score(y_test, y_pred, average='binary')
    print('Recall: %.3f' % recall)

    """
    print(confusion_matrix(labels, y_pred))
    precision = precision_score(labels, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    recall = recall_score(labels, y_pred, average='binary')
    print('Recall: %.3f' % recall)
    """
    pass

def createAttackerRecord(filename): # extract malicious vehicles from given csv
    df = pd.read_csv(filename)
    attacker_dict = get_attackers(df)
    json_name = os.path.splitext(filename)[0] + "_attackers.json"
    with open(json_name, 'w') as fp:
        json.dump(attacker_dict, fp)
    return json_name # return the name of dumped file

if __name__ == "__main__":

    # find better extractors

    # case 1
    filename = "veins_maat.uc1.14505247.180205_165727.csv"
    print(filename)
    attacker_filename = "veins_maat.uc1.14505247.180205_165727_attackers.json"  # createAttackerRecord(filename)

    feature_file_name,label_file_name= featureExtractorFromcsv(filename,attacker_filename,full_set=True)
    features_file = feature_file_name # "veins_maat.uc1.14505247.180205_165727_features.csv"  # feature_file_name
    labels_file = label_file_name # "veins_maat.uc1.14505247.180205_165727_labels.csv"  # label_file_name
    makePrediction(features_file, labels_file)



    exit()


    # case 1
    filename = "veins_maat.uc1.14505247.180205_165727.csv"
    print(filename)
    attacker_filename = "veins_maat.uc1.14505247.180205_165727_attackers.json" #createAttackerRecord(filename)

    #feature_file_name,label_file_name= featureExtractorFromcsv(filename,attacker_filename)
    features_file = "veins_maat.uc1.14505247.180205_165727_features.csv" #feature_file_name
    labels_file = "veins_maat.uc1.14505247.180205_165727_labels.csv" #label_file_name
    makePrediction(features_file,labels_file)

    # case 2
    filename = "veins_maat.uc1.14505342.180205_171010.csv"
    print(filename)
    attacker_filename ="veins_maat.uc1.14505342.180205_171010_attackers.json" #createAttackerRecord(filename)

    #feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file ="veins_maat.uc1.14505342.180205_171010_features.csv" #feature_file_name
    labels_file = "veins_maat.uc1.14505342.180205_171010_labels.csv" #label_file_name
    makePrediction(features_file, labels_file)



    # case 3
    filename = "veins_maat.uc1.14505511.180205_173553.csv"
    print(filename)
    attacker_filename = "veins_maat.uc1.14505511.180205_173553_attackers.json"#createAttackerRecord(filename)

    #feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file = "veins_maat.uc1.14505511.180205_173553_features.csv"#feature_file_name
    labels_file = "veins_maat.uc1.14505511.180205_173553_labels.csv"#label_file_name
    makePrediction(features_file, labels_file)

    # case 4
    filename = "veins_maat.uc1.14505930.180205_192941.csv"
    print(filename)
    attacker_filename = "veins_maat.uc1.14505930.180205_192941_attackers.json" #createAttackerRecord(filename)

    #feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file = "veins_maat.uc1.14505930.180205_192941_features.csv" #feature_file_name
    labels_file = "veins_maat.uc1.14505930.180205_192941_labels.csv"#label_file_name
    makePrediction(features_file, labels_file)


    # case 5
    filename = "veins_maat.uc1.14506016.180205_200240.csv"
    print(filename)
    attacker_filename ="veins_maat.uc1.14506016.180205_200240_attackers.json" #createAttackerRecord(filename)

    #feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file ="veins_maat.uc1.14506016.180205_200240_features.csv" #feature_file_name
    labels_file ="veins_maat.uc1.14506016.180205_200240_labels.csv" #label_file_name
    makePrediction(features_file, labels_file)



    exit()


    # case 3
    filename = "veins_maat.uc1.14505511.180205_173553.csv"
    print(filename)
    attacker_filename = createAttackerRecord(filename)

    feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file = feature_file_name
    labels_file = label_file_name
    makePrediction(features_file, labels_file)

    # case 4
    filename = "veins_maat.uc1.14505930.180205_192941.csv"
    print(filename)
    attacker_filename = createAttackerRecord(filename)

    feature_file_name, label_file_name = featureExtractorFromcsv(filename, attacker_filename)
    features_file = feature_file_name
    labels_file = label_file_name
    makePrediction(features_file, labels_file)

    print("ALL DONE")
    """
    print("Start extracting features from  time series")
    df = pd.read_csv("veins_maat.uc1.14505247.180205_165727.csv")
    features_filtered,labels = extractFeatures(df)
    print("done feature extraction")
    features_filtered.to_csv('features.csv')
    savetxt('labels.csv', labels, delimiter=',')
    print("Done")
    
    
    
    | Repetition | Density Level | Attacker Type | Attacker Density | Name/Link |
    | --- | --- | --- | --- | --- |
    | 0 | 7 | 1 | 0.3 | [veins_maat.uc1.14505247.180205_165727.tgz](https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/veins_maat.uc1.14505247.180205_165727.tgz) |
    | 0 | 7 | 2 | 0.3 | [veins_maat.uc1.14505342.180205_171010.tgz](https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/veins_maat.uc1.14505342.180205_171010.tgz) |
    | 0 | 7 | 4 | 0.3 | [veins_maat.uc1.14505511.180205_173553.tgz](https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/veins_maat.uc1.14505511.180205_173553.tgz) |
    | 0 | 7 | 8 | 0.3 | [veins_maat.uc1.14505930.180205_192941.tgz](https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/veins_maat.uc1.14505930.180205_192941.tgz) |
    | 0 | 7 | 16 | 0.3 | [veins_maat.uc1.14506016.180205_200240.tgz](https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/veins_maat.uc1.14506016.180205_200240.tgz) |
    """


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
