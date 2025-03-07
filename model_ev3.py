from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

class WasteTracker:
    def __init__(self, model_path, distance_threshold, pair_consistency_threshold, max_inactive_frames, temporal_window=10, temporal_threshold=7):
        self.model = YOLO(model_path)
        self.distance_threshold = distance_threshold
        self.pair_consistency_threshold = pair_consistency_threshold
        self.max_inactive_frames = max_inactive_frames
        self.temporal_window = temporal_window
        self.temporal_threshold = temporal_threshold
        self.tracking_data = {}
        self.next_id = 1

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
                self.tracking_data[self.next_id] = {
                    'class_id': det['class_id'],
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
                    'last_seen': current_frame,
                    'proximity_buffer': deque([0] * self.temporal_window, maxlen=self.temporal_window)  # Initialize buffer
                }
                det['id'] = self.next_id
                self.next_id += 1
            return

        if not detections:
            return

        distance_matrix = self.compute_distance_matrix(detections, self.tracking_data)

        if distance_matrix.size == 0:
            for det in detections:
                det['id'] = self.next_id
                self.tracking_data[self.next_id] = {
                    'class_id': det['class_id'],
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
                    'last_seen': current_frame,
                    'proximity_buffer': deque([0] * self.temporal_window, maxlen=self.temporal_window)
                }
                self.next_id += 1
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
                    self.tracking_data[track_id]['coordinates'].append(self.calculate_center(det['bbox']))
                    self.tracking_data[track_id]['last_seen'] = current_frame
                    assigned_ids.add(track_id)

        for det in detections:
            if 'id' not in det:
                det['id'] = self.next_id
                self.tracking_data[self.next_id] = {
                    'class_id': det['class_id'],
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
                    'last_seen': current_frame,
                    'proximity_buffer': deque([0] * self.temporal_window, maxlen=self.temporal_window)
                }
                self.next_id += 1

        self.clean_inactive_tracks(current_frame)

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

    def identify_temporal_pairs(self, current_frame):
        humans = {track_id: data for track_id, data in self.tracking_data.items() if data['class_id'] == 0}
        trash = {track_id: data for track_id, data in self.tracking_data.items() if data['class_id'] == 1}

        temporal_pairs = []
        for human_id, human_data in humans.items():
            human_center = human_data['coordinates'][-1]
            is_trash_near = False

            for trash_id, trash_data in trash.items():
                trash_center = trash_data['coordinates'][-1]
                distance = self.compute_distance(human_center, trash_center)

                if distance < self.distance_threshold:
                    is_trash_near = True
                    break  # Only need one trash nearby to increment the buffer

            # Update temporal buffer for the human
            human_data['proximity_buffer'].append(1 if is_trash_near else 0)

            # Check if trash is consistently near
            if sum(human_data['proximity_buffer']) >= self.temporal_threshold:
                temporal_pairs.append(human_id)

        return temporal_pairs

    def clean_inactive_tracks(self, current_frame):
        inactive_tracks = []
        for track_id, data in self.tracking_data.items():
            if current_frame - data['last_seen'] > self.max_inactive_frames:
                inactive_tracks.append(track_id)

        for track_id in inactive_tracks:
            del self.tracking_data[track_id]

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        display_size = (frame_width, frame_height)

        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = []
            results = self.model(frame)
            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id in [0, 1]:  # Human: 0, Trash: 1
                        detections.append({'bbox': bbox, 'class_id': class_id})

            self.assign_ids(detections, current_frame)
            temporal_pairs = self.identify_temporal_pairs(current_frame)

            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = (0, 255, 0) if det['class_id'] == 1 else (0, 0, 255)
                label = f"ID {det['id']} - {'Trash' if det['class_id'] == 1 else 'Human'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for human_id in temporal_pairs:
                human_center = self.tracking_data[human_id]['coordinates'][-1]
                cv2.putText(frame, f"Suspected: ID {human_id}", human_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            resized_frame = cv2.resize(frame, display_size)
            cv2.imshow("Tracking", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            current_frame += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\100epoch.pt"
    video_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\1styrmech.mp4"
    tracker = WasteTracker(
        model_path=model_path, 
        distance_threshold=50, 
        pair_consistency_threshold=5, 
        max_inactive_frames=15,
        temporal_window=10,
        temporal_threshold=7
    )
    tracker.process_video(video_path)
