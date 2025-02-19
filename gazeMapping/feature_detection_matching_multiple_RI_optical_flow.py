import cv2
import numpy as np
import pandas as pd
import os
import torch
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout

# define variables
path_to_recordings = 'data/referenceImages'
path_to_gazedata_file = 'gazedata_ivt.xlsx'
path_to_video = 'scenevideo.mp4'
recordings = os.listdir(path_to_recordings)
path_to_reference_images = 'data/referenceImages'
output_file = 'output.txt'
flow_threshold = 3.0

# Configuration for SuperPoint and SuperGlue
superpoint_config = {
    'nms_radius': 6,
    'keypoint_threshold': 0.01,
    'max_keypoints': 1024
}

superglue_config = {
    'weights': 'outdoor',
    'sinkhorn_iterations': 50,
    'match_threshold': 0.1,
}


with open(output_file, 'w') as file:
    with redirect_stdout(file):
        # Initialize SuperPoint and SuperGlue
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        superpoint = SuperPoint(superpoint_config).to(device)
        superglue = SuperGlue(superglue_config).to(device)

        # Helper function to process images with SuperPoint
        def extract_superpoint_features(image):
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
            
            image_tensor = torch.from_numpy(gray_image).float().unsqueeze(0).unsqueeze(0) / 255.
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                output = superpoint({'image': image_tensor})
                keypoints = output['keypoints'][0].cpu().numpy()
                descriptors = output['descriptors'][0].cpu().numpy()
                scores = output['scores'][0].cpu().numpy()
            return keypoints, descriptors, scores
        
        def process_gazedata(path_to_r):
            gazedata_df = pd.read_excel(os.path.join(path_to_r, path_to_gazedata_file))
            time = gazedata_df['Recording timestamp'].to_numpy() # in seconds
            participant = gazedata_df['Participant name'].to_numpy()
            condition = gazedata_df['Condition'].to_numpy()
            gaze_x = gazedata_df['Gaze point X'].to_numpy()
            gaze_y = gazedata_df['Gaze point Y'].to_numpy()
            movement_type = gazedata_df['Movement Type'].to_numpy()
            
            return time, condition, participant, gaze_x, gaze_y, movement_type
        
        def process_reference_images(curr_condition, path_to_reference_images):
            path_to_curr_reference_images = os.path.join(path_to_reference_images, curr_condition)
            images_files = [file for file in os.listdir(path_to_curr_reference_images) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            reference_images = []
            for i in images_files:
                reference_images.append(os.path.join(path_to_curr_reference_images, i))
            
            reference_features = []
            for reference_image_path in reference_images:
                # Load the reference image
                reference_image = cv2.imread(reference_image_path)
                if reference_image is None:
                    print(f"Error: Could not load reference image from {reference_image_path}")
                    continue
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                reference_image_h, reference_image_w = reference_image.shape[:2]
                # Extract features using SuperPoint
                ref_keypoints, ref_descriptors, ref_scores = extract_superpoint_features(reference_image)                
                # Store the extracted features in a list for later comparison
                reference_features.append({
                    "image_path": reference_image_path,
                    "keypoints": ref_keypoints,
                    "descriptors": ref_descriptors,
                    "scores": ref_scores,
                    "dimensions": (reference_image_w, reference_image_h)
                })
            print(f"lent reference_features: {len(reference_features)}")
            return reference_features
        
        # Function to calculate optical flow using Lucas-Kanade method
        def calculate_optical_flow(prev_frame, next_frame):
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if prev_pts is None:
                print("No good features found, skipping frame.")
                return np.zeros_like(prev_frame)
            # Calculate optical flow using Lucas-Kanade
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, **lk_params)
            # Create a mask image for drawing purposes
            flow = np.zeros_like(prev_frame)
            # Select good points
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]
            # Draw the flow vectors as arrows
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw an arrow between the points
                flow = cv2.arrowedLine(flow, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

            return flow        

        def run_matching(path_to_r):
            columns = [
                'Timestamp', 
                'Frame index',
                'Gaze index',
                'Participant name', 
                'Condition', 
                'Gaze point X', 
                'Gaze point Y', 
                'Unnorm Gaze point X',
                'Unnorm Gaze point Y',
                'Reference Image', 
                'Reference Image Width', 
                'Reference Image Height',
                'Mapped Gaze point X', 
                'Mapped Gaze point Y',
                'Optical flow avg magnitude',        
            ]
            mapped_gaze_points_df = pd.DataFrame(columns=columns)
            
            time, condition, participant, gaze_x, gaze_y, movement_type = process_gazedata(path_to_r)
            curr_participant = participant[1]
            curr_condition = condition[1]
            
            reference_features = process_reference_images(curr_condition, path_to_reference_images)

            video = os.path.join(path_to_r, path_to_video)
            cap = cv2.VideoCapture(video) 
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)         

            start_frame = 2900
            end_frame = 5000
            prev_avg_magnitude = 0
            start_frame_ts = start_frame / fps

            if start_frame == 0:
                gaze_index = 0
            else:
                gaze_index = 0
                while gaze_index < len(time) and time[gaze_index] < start_frame_ts:
                    gaze_index += 1 
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, curr_frame = cap.read() 
            curr_frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 
            frame_count = start_frame

            while cap.isOpened() and (frame_count <= end_frame):
                if not ret:
                    break
                if gaze_index >= len(time):
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                curr_frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                next_frame_ts = (frame_count + 1) /fps
                                
                curr_frame_gaze_data = []
                
                while gaze_index < len(time) and time[gaze_index] < next_frame_ts:
                    if movement_type[gaze_index] == "Fixation" and not (int(gaze_x[gaze_index]) == -1) and not (int(gaze_y[gaze_index]) == -1):
                        curr_frame_gaze_data.append({
                            'Gaze index': gaze_index,
                            'Gaze timestamp': time[gaze_index],
                            'Gaze Point X': gaze_x[gaze_index],
                            'Gaze Point Y': gaze_y[gaze_index],
                            'Movement Type': movement_type[gaze_index]
                        })  
                        gaze_index += 1
                    else:
                        gaze_index += 1

                    for gaze_point in curr_frame_gaze_data:
                        gaze_point_timestamp = {gaze_point['Gaze timestamp']}
                        gaze_point_index = gaze_point['Gaze index']
                        gaze_point_x = gaze_point['Gaze Point X']
                        gaze_point_y = gaze_point['Gaze Point Y']
                        normalized_point = np.array([gaze_point_x, gaze_point_y]) 
                        unnorm_gaze_point_x = gaze_point_x * frame_w
                        unnorm_gaze_point_y = gaze_point_y * frame_h
                        unnormalized_point = np.array([unnorm_gaze_point_x, unnorm_gaze_point_y])
                        frame_image = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                        frame_keypoints, frame_descriptors, frame_scores = extract_superpoint_features(frame_image)
                        
                        frame_keypoints_number = len(frame_keypoints)
                        frame_descriptors_number = frame_descriptors.shape[0]
                        frame_scores_shape = frame_scores.shape
                        
                        score_threshold = 0.5
                        valid_keypoints = [keypoint for i, keypoint in enumerate(frame_keypoints) if frame_scores[i] >= score_threshold]
                        
                        best_inlier_ratio = -1 
                        best_H = None
                        best_reference_image = None
                        best_match_image = None
                        best_point_in_reference = None
                        for ref_feature in reference_features:
                            ref_keypoints = ref_feature['keypoints']
                            ref_descriptors = ref_feature['descriptors']
                            ref_scores = ref_feature['scores']
                            reference_image = cv2.imread(ref_feature['image_path'])
                            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                            reference_image_h, reference_image_w = reference_image.shape[:2]
                            
                            data = {
                                'keypoints0': torch.from_numpy(ref_keypoints).float().unsqueeze(0).to(device),  # Reference image keypoints (target)
                                'keypoints1': torch.from_numpy(frame_keypoints).float().unsqueeze(0).to(device),  # Frame keypoints (source)
                                'descriptors0': torch.from_numpy(ref_descriptors).float().unsqueeze(0).to(device),  # Reference image descriptors (target)
                                'descriptors1': torch.from_numpy(frame_descriptors).float().unsqueeze(0).to(device),  # Frame descriptors (source)
                                'scores0': torch.from_numpy(ref_scores).float().unsqueeze(0).to(device),  # Reference image scores (target)
                                'scores1': torch.from_numpy(frame_scores).float().unsqueeze(0).to(device),  # Frame scores (source)
                                'image0': torch.empty((1, 1) + reference_image.shape[:2]).to(device),  # Dummy image tensor for reference image (target)
                                'image1': torch.empty((1, 1) + frame_image.shape[:2]).to(device)  # Dummy image tensor for frame image (source)
                            }
                        
                            with torch.no_grad():
                                matching_output = superglue(data)
                                matches = matching_output['matches0'][0].cpu().numpy()
                                matching_confidence = matching_output['matching_scores0'][0].cpu().numpy() 

                            matching_confidence_threshold = 0.7
                            valid_matches = matching_confidence >= matching_confidence_threshold 
                            frame_matched_kp = frame_keypoints[matches[valid_matches]] 
                            ref_matched_kp = ref_keypoints[valid_matches]            
                            num_matches = len(ref_matched_kp)

                            print(f"Num_matches: {num_matches}")
                            if num_matches >= 4:
                                H, inliers = cv2.findHomography(frame_matched_kp, ref_matched_kp, cv2.RANSAC, ransacReprojThreshold=5.0)#, maxIters=10000) # lower RANSAC threshold allows more precise inlier matches 
                                
                                inliers = inliers.flatten()
                                num_inliers = np.sum(inliers)
                                inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
                                
                                if inlier_ratio < 0.5:  # If less than 50% of the matches are inliers, discard the homography
                                    print("Not enough inliers, skipping this candidate image.")
                                    #continue
                                else:
                                    print(f"Homography has {inlier_ratio:.2f} inliers.")
                                
                                if inlier_ratio > best_inlier_ratio:
                                    best_inlier_ratio = inlier_ratio
                                    best_H = H
                                    best_reference_image = reference_image
                                    best_ref_feature = ref_feature
                                    best_match_image = cv2.drawMatches(
                                        frame_image, [cv2.KeyPoint(x, y, 1) for x, y in frame_matched_kp],
                                        reference_image, [cv2.KeyPoint(x, y, 1) for x, y in ref_matched_kp],
                                        [cv2.DMatch(i, i, 0) for i in range(num_matches)], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                    )
                                    print(f"unnormalized point before H: {unnormalized_point}")
                                    best_point_in_reference = best_H @ np.append(unnormalized_point, 1).reshape(3, 1)
                                    best_point_in_reference /= best_point_in_reference[2]  # Normalize to get (x, y)
                                    best_point_in_reference = best_point_in_reference[:2].flatten()
                                    print(f"best point in reference: {best_point_in_reference}")
                            
                        print(best_ref_feature['image_path'])
                        if best_reference_image is not None:
                            new_row = {
                                'Timestamp': list(gaze_point_timestamp)[0],
                                'Frame index': frame_count,
                                'Gaze index': gaze_point_index,
                                'Participant name': curr_participant,
                                'Condition': curr_condition,
                                'Gaze point X': gaze_point_x,
                                'Gaze point Y': gaze_point_y,
                                'Unnorm Gaze point X': unnorm_gaze_point_x,
                                'Unnorm Gaze point Y': unnorm_gaze_point_y,
                                'Reference Image': best_ref_feature['image_path'],
                                'Reference Image Width': best_ref_feature['dimensions'][0],
                                'Reference Image Height': best_ref_feature['dimensions'][1],
                                'Mapped Gaze point X': best_point_in_reference[0],
                                'Mapped Gaze point Y': best_point_in_reference[1],
                                'Optical flow avg magnitude': prev_avg_magnitude,
                            }
                            
                            mapped_gaze_points_df = pd.concat([mapped_gaze_points_df, pd.DataFrame([new_row])], ignore_index=True)
                            output_excel_path = os.path.join(path_to_r, 'mapped_gaze_points.xlsx')
                            mapped_gaze_points_df.to_excel(output_excel_path, index=False)
                        else:
                            print(f"Frame {gaze_point_index}: No valid matches found for any reference image.")
                
                ret, next_frame = cap.read()
                flow = calculate_optical_flow(curr_frame, next_frame)  
                
                flow_magnitude = np.linalg.norm(flow, axis=2)
                avg_magnitude = np.mean(flow_magnitude)

                curr_frame = next_frame
                prev_avg_magnitude = avg_magnitude                
                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()
            
            output_excel_path = os.path.join(path_to_r, 'mapped_gaze_points.xlsx')
            mapped_gaze_points_df.to_excel(output_excel_path, index=False)
            print(f"Mapped gaze points saved to {output_excel_path}")
                            
        # loop over all recordings and run the matching
        for r in recordings:
            # get path to one recording folder
            path_to_r = os.path.join(path_to_recordings, r)
            run_matching(path_to_r)
