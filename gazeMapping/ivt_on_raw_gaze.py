import os
import json
import gzip
import pandas as pd
import datetime
import cv2
import numpy as np

path_to_recordings = 'data/rawRecordings'

def get_participant_name(dir):
    participant_meta = os.path.join(dir, 'meta', 'participant') 
    try:
        with open(participant_meta, 'r') as f:
            data = json.load(f)
        participant_name = data['name']
        return participant_name
    except FileNotFoundError:
        print(f"Error: The file {participant_meta} does not exist.")
        return None
    
def get_condition(participant):
    if participant is None:
        print("Error: Participant name is None, cannot extract condition.")
        return None
    p = participant.split('_')
    cond = p[1]
    return cond

def calculate_velocity(gaze_x, gaze_y, timestamps, screen_width, screen_height):
    velocities = []
    for i in range(1, len(gaze_x)):
        dx = gaze_x[i] - gaze_x[i-1]
        dy = gaze_y[i] - gaze_y[i-1]
        dt = timestamps[i] - timestamps[i-1]
        
        angular_velocity = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
        angular_velocity_deg_per_sec = angular_velocity * 180 / np.pi 
        velocities.append(angular_velocity_deg_per_sec)
    return velocities

def apply_ivt(velocities, threshold=100):
    events = []
    for velocity in velocities:
        if velocity < threshold:
            events.append("Fixation")
        else:
            events.append("Saccade")
    return events

def get_gazedata(recording):
    
    participant = get_participant_name(recording)
    if participant is None:
        return
    
    cond = get_condition(participant)
    
    gazedata = os.path.join(recording, 'gazedata.gz')  # Ensure correct path format
    if not os.path.isfile(gazedata):
        print(f'File: {gazedata} does not exist.')
        return
    
    gazedata_list = []
    
    timestamp = []
    data = []
    gaze2d_x = []
    gaze2d_y = []
    participant_name = []
    condition = []
    
    gaze_df = pd.DataFrame()
    
    with gzip.open(gazedata, 'r') as f_in:
        for line in f_in:
            temp = line.decode('utf-8')
            gazedata_list.append(temp)
        
        gazedata_json = [json.loads(g) for g in gazedata_list]
        
        timestamp = [g['timestamp'] for g in gazedata_json] 
        data = [g['data'] for g in gazedata_json]
        
        gaze2d_x = [g['gaze2d'][0] if 'gaze2d' in g else -1 for g in data]
        gaze2d_y = [g['gaze2d'][1] if 'gaze2d' in g else -1 for g in data]
        
        length = len(timestamp)
        
        for l in range(length):
            participant_name.append(participant)
            condition.append(cond)
            
        gaze_df['Recording timestamp'] = timestamp
        gaze_df['Participant name'] = participant_name
        gaze_df['Condition'] = condition
        gaze_df['Gaze point X'] = gaze2d_x
        gaze_df['Gaze point Y'] = gaze2d_y
        
        screen_width, screen_height = 1920, 1080 
        velocities = calculate_velocity(gaze2d_x, gaze2d_y, timestamp, screen_width, screen_height)
        
        ivt_labels = apply_ivt(velocities, threshold=100) 
        gaze_df['Movement Type'] = ['No Data'] + ivt_labels
        gaze_df.to_excel(os.path.join(recording, 'gazedata_ivt.xlsx'), index=False)
    
    return gaze_df

recordings = os.listdir(path_to_recordings)
for r in recordings:
    print(os.path.join(path_to_recordings, r))
    get_gazedata(os.path.join(path_to_recordings, r))
