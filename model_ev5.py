from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import os

class WasteTracker:
    def __init__(self, model_path, evidence_path, distance_threshold, pair_consistency_threshold, max_inactive_frames, 
                 temporal_window=10, temporal_threshold=7, min_holding_frames=15, min_disposal_frames=20):
        self.model = YOLO(model_path)
        self.evidence_path = evidence_path
        self.distance_threshold = distance_threshold
        self.pair_consistency_threshold = pair_consistency_threshold
        self.max_inactive_frames = max_inactive_frames
        self.temporal_window = temporal_window
        self.temporal_threshold = temporal_threshold
        self.min_holding_frames = min_holding_frames
        self.min_disposal_frames = min_disposal_frames
        self.tracking_data = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Create the main evidence directory if it doesn't exist
        os.makedirs(self.evidence_path, exist_ok=True)

    def calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return int(center_x), int(center_y)

    def compute_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def assign_ids(self, detections, current_frame):
        if not self.tracking_data:
            for det in detections:
                self.initialize_track(det, current_frame)
            return

        if not detections:
            return

        distance_matrix = self.compute_distance_matrix(detections, self.tracking_data)

        if distance_matrix.size == 0:
            for det in detections:
                self.initialize_track(det, current_frame)
            return

        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        assigned_ids = set()
        for row, col in zip(row_indices, col_indices):
            if col >= len(self.tracking_data) or row >= len(detections):
                continue

            if distance_matrix[row, col] < self.distance_threshold:
                det = detections[row]
                track_id = list(self.tracking_data.keys())[col]
                if track_id not in assigned_ids:
                    det['id'] = track_id
                    self.update_track(track_id, det, current_frame)
                    assigned_ids.add(track_id)

        for det in detections:
            if 'id' not in det:
                self.initialize_track(det, current_frame)

        self.clean_inactive_tracks(current_frame)

    def initialize_track(self, det, current_frame):
        track_id = self.next_id
        self.tracking_data[track_id] = {
            'class_id': det['class_id'],
            'bbox': det['bbox'],
            'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
            'last_seen': current_frame,
            'proximity_buffer': deque([0] * self.temporal_window, maxlen=self.temporal_window),
            'frame_buffer': deque(maxlen=10),
            'state': 'IDLE',
            'no_trash_count': 0
        }
        det['id'] = track_id
        self.next_id += 1

    def update_track(self, track_id, det, current_frame):
        self.tracking_data[track_id]['bbox'] = det['bbox']
        self.tracking_data[track_id]['coordinates'].append(self.calculate_center(det['bbox']))
        self.tracking_data[track_id]['last_seen'] = current_frame

    def compute_distance_matrix(self, detections, tracking_data):
        distance_matrix = []
        for det in detections:
            det_center = self.calculate_center(det['bbox'])
            distances = []
            for track in tracking_data.values():
                track_center = track['coordinates'][-1]
                distance = self.compute_distance(det_center, track_center)
                distances.append(distance)
            distance_matrix.append(distances)
        return np.array(distance_matrix)

    def process_proximity_logic(self, detections, frame):
        for track_id, track_data in self.tracking_data.items():
            if track_data['class_id'] != 0:  # Only process humans
                continue

            human_center = track_data['coordinates'][-1]
            is_trash_near = False
            human_id = track_id

            for det in detections:
                if det['class_id'] == 1:  # Trash
                    trash_center = self.calculate_center(det['bbox'])
                    distance = self.compute_distance(human_center, trash_center)
                    if distance < self.distance_threshold:
                        is_trash_near = True
                        break

            track_data['proximity_buffer'].append(1 if is_trash_near else 0)

            if self.frame_count % 4 == 0:
                track_data['frame_buffer'].append(frame.copy())

            consecutive_trash_frames = sum(track_data['proximity_buffer'])

            if track_data['state'] == 'IDLE':
                if consecutive_trash_frames >= self.min_holding_frames:
                    track_data['state'] = 'HOLDING_TRASH'
                    track_data['no_trash_count'] = 0
                    print(f"Human {human_id} confirmed to be holding trash after {consecutive_trash_frames} frames")
            
            elif track_data['state'] == 'HOLDING_TRASH':
                if is_trash_near:
                    track_data['no_trash_count'] = 0
                else:
                    track_data['no_trash_count'] += 1
                    
                    if track_data['no_trash_count'] >= self.min_disposal_frames:
                        track_data['state'] = 'TRASH_DISPOSED'
                        print(f"Human {human_id} disposed trash after {track_data['no_trash_count']} frames without trash")
                        self.save_evidence(human_id, track_data['frame_buffer'], frame)
                        track_data['frame_buffer'].clear()
                        track_data['proximity_buffer'] = deque([0] * self.temporal_window, maxlen=self.temporal_window)
                        track_data['no_trash_count'] = 0

    def save_evidence(self, human_id, frame_buffer, current_frame):
        # Create a subfolder for this specific human ID within the evidence path
        human_folder = os.path.join(self.evidence_path, f"human_{human_id}")
        os.makedirs(human_folder, exist_ok=True)
        print(f"Saving evidence for Human ID {human_id} in {human_folder}")

        for idx, frame in enumerate(frame_buffer):
            x1, y1, x2, y2 = map(int, self.tracking_data[human_id]['bbox'])
            
            frame_clone = frame.copy()
            cv2.rectangle(frame_clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_clone, f"Human ID {human_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(human_folder, f"frame_{idx:03d}.jpg"), frame_clone)

        current_frame_clone = current_frame.copy()
        x1, y1, x2, y2 = map(int, self.tracking_data[human_id]['bbox'])
        cv2.rectangle(current_frame_clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(current_frame_clone, f"Human ID {human_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(human_folder, "final_frame.jpg"), current_frame_clone)

    def clean_inactive_tracks(self, current_frame):
        inactive_tracks = [track_id for track_id, data in self.tracking_data.items()
                         if current_frame - data['last_seen'] > self.max_inactive_frames]
        for track_id in inactive_tracks:
            del self.tracking_data[track_id]

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            new_width = int(frame.shape[1])
            new_height = int(frame.shape[0])
            frame = cv2.resize(frame, (new_width, new_height))
            
            detections = []
            results = self.model(frame, conf=0.5, iou=0.4)
            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id in [0, 1, 2]:  # Human: 0, Trash: 1, vehicle: 2
                        detections.append({'bbox': bbox, 'class_id': class_id})

            self.assign_ids(detections, current_frame)
            self.process_proximity_logic(detections, frame)

            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = (0, 255, 0) if det['class_id'] == 1 else (0, 0, 255)
                label = f"ID {det['id']} - {'Trash' if det['class_id'] == 1 else 'Human'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            y_offset = 30
            for track_id, track_data in self.tracking_data.items():
                if track_data['class_id'] == 0:  # Only display human states
                    state = track_data['state']
                    display_text = f"Human ID {track_id}: {state}"
                    cv2.putText(frame, display_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            current_frame += 1
            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\100epochv2.pt"
    video_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\1styrmech.mp4"
    evidence_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\evidence"  # Specify your evidence folder path here
    
    tracker = WasteTracker(
        model_path=model_path,
        evidence_path=evidence_path,  # Added evidence_path parameter
        distance_threshold=150,
        pair_consistency_threshold=10,
        max_inactive_frames=90,
        temporal_window=10,
        temporal_threshold=7,
        min_holding_frames=5,
        min_disposal_frames=10
    )
    tracker.process_video(video_path)