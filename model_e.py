import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from collections import deque

class WasteTracker:
    def __init__(self, model_path, distance_threshold, pair_consistency_threshold, max_inactive_frames):
        self.model = YOLO(model_path)
        self.distance_threshold = distance_threshold
        self.pair_consistency_threshold = pair_consistency_threshold
        self.tracking_data = {}
        self.next_id = 1
        self.pair_counters = {}
        self.max_inactive_frames = max_inactive_frames
        self.inactive_tracks = {}

    def calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return int(center_x), int(center_y)

    def compute_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def assign_ids(self, detections, current_frame):
        if not self.tracking_data:  # No previous tracks
            for det in detections:
                self.tracking_data[self.next_id] = {
                    'class_id': det['class_id'],
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
                    'last_seen': current_frame
                }
                det['id'] = self.next_id
                self.next_id += 1
            return

        if not detections:  # No detections in the current frame
            return

        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(detections, self.tracking_data)

        # Handle cases where one dimension is empty
        if distance_matrix.size == 0:
            for det in detections:
                det['id'] = self.next_id
                self.tracking_data[self.next_id] = {
                    'class_id': det['class_id'],
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=10),
                    'last_seen': current_frame
                }
                self.next_id += 1
            return

        # Solve the assignment problem
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        assigned_ids = set()
        for row, col in zip(row_indices, col_indices):
            if col >= len(self.tracking_data) or row >= len(detections):
                continue  # Ignore invalid indices

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
                    'coordinates': deque([self.calculate_center(det['bbox'])], maxlen=5),
                    'last_seen': current_frame
                }
                self.next_id += 1

        # Remove inactive tracks
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

    def identify_pairs(self, current_frame):
        humans = {track_id: data for track_id, data in self.tracking_data.items() if data['class_id'] == 0}
        trash = {track_id: data for track_id, data in self.tracking_data.items() if data['class_id'] == 1}

        pairs = []
        for human_id, human_data in humans.items():
            human_center = human_data['coordinates'][-1]
            closest_trash_id = None
            closest_distance = self.distance_threshold

            for trash_id, trash_data in trash.items():
                trash_center = trash_data['coordinates'][-1]
                distance = self.compute_distance(human_center, trash_center)

                if distance < closest_distance:
                    closest_trash_id = trash_id
                    closest_distance = distance

            if closest_trash_id:
                pair = (human_id, closest_trash_id)
                if pair not in self.pair_counters:
                    self.pair_counters[pair] = current_frame
                else:
                    self.pair_counters[pair] = current_frame

                if current_frame - self.pair_counters[pair] < self.pair_consistency_threshold:
                    pairs.append(pair)

        # Remove stale pairs
        self.pair_counters = {pair: frame for pair, frame in self.pair_counters.items() if current_frame - frame <= 15}

        return pairs

    def clean_inactive_tracks(self, current_frame):
        inactive_tracks = []
        for track_id, data in self.tracking_data.items():
            if current_frame - data['last_seen'] > self.max_inactive_frames:
                inactive_tracks.append(track_id)

        for track_id in inactive_tracks:
            del self.tracking_data[track_id]

    def draw_tracking(self, frame):
        for track_id, data in self.tracking_data.items():
            color = (0, 255, 0) if data['class_id'] == 1 else (0, 0, 255)

    def draw_pairs(self, frame, pairs):
        for human_id, trash_id in pairs:
            human_center = self.tracking_data[human_id]['coordinates'][-1]
            trash_center = self.tracking_data[trash_id]['coordinates'][-1]
            cv2.line(frame, human_center, trash_center, (0, 255, 255), 2)
            label = f"Human ID {human_id} -- Trash ID {trash_id}"
            mid_point = ((human_center[0] + trash_center[0]) // 2, (human_center[1] + trash_center[1]) // 2)
            cv2.putText(frame, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
                    if class_id in [0, 1]:
                        detections.append({'bbox': bbox, 'class_id': class_id})

            self.assign_ids(detections, current_frame)
            pairs = self.identify_pairs(current_frame)

            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color = (0, 255, 0) if det['class_id'] == 1 else (0, 0, 255)
                label = f"ID {det['id']} - {'Trash' if det['class_id'] == 1 else 'Human'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.draw_tracking(frame)
            self.draw_pairs(frame, pairs)

            resized_frame = cv2.resize(frame, display_size)
            cv2.imshow("Tracking", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            current_frame += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\100epoch.pt"
    video_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\civil-backside.mp4"
    tracker = WasteTracker(model_path=model_path, distance_threshold=50, pair_consistency_threshold=5, max_inactive_frames=15)
    tracker.process_video(video_path)
