import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Helper function to calculate the center of a bounding box
def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)

# Compute the distance matrix between detections and previous tracks
def compute_distance_matrix(detections, previous_tracks):
    distance_matrix = []
    for det in detections:
        det_center = calculate_center(det)
        distances = []
        for track_id, track_data in previous_tracks.items():
            track_center = track_data["center"]
            distance = np.sqrt((det_center[0] - track_center[0])**2 +
                               (det_center[1] - track_center[1])**2)
            distances.append(distance)
        distance_matrix.append(distances)
    return np.array(distance_matrix)

# Match detections to tracks using Hungarian algorithm
def match_detections_to_tracks(detections, previous_tracks, threshold=50):
    if not detections or not previous_tracks:
        updated_tracks = {}
        new_id = max(previous_tracks.keys(), default=0) + 1
        for i, det in enumerate(detections):
            updated_tracks[new_id] = {
                "bbox": det,
                "center": calculate_center(det),
                "path": [calculate_center(det)]
            }
            new_id += 1
        return updated_tracks

    distance_matrix = compute_distance_matrix(detections, previous_tracks)
    if distance_matrix.ndim == 1:
        distance_matrix = distance_matrix[:, np.newaxis]

    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    updated_tracks = {}
    assigned_detections = set()

    for row, col in zip(row_indices, col_indices):
        if distance_matrix[row, col] < threshold:
            track_id = list(previous_tracks.keys())[col]
            updated_tracks[track_id] = {
                "bbox": detections[row],
                "center": calculate_center(detections[row]),
                "path": previous_tracks[track_id]["path"] + [calculate_center(detections[row])]
            }
            assigned_detections.add(row)

    new_id = max(previous_tracks.keys(), default=0) + 1
    for i, det in enumerate(detections):
        if i not in assigned_detections:
            updated_tracks[new_id] = {
                "bbox": det,
                "center": calculate_center(det),
                "path": [calculate_center(det)]
            }
            new_id += 1

    return updated_tracks

def main():
    model = YOLO(r'C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\runs\detect\run13\weights\best.pt')
    input_video_path = r"C:\Users\DELL\Desktop\trash\veh_pavi\vehidet\video2.mp4"
    cap = cv2.VideoCapture(input_video_path)

    tracking_data = {
        "bus": {}, "car": {}, "motorbike": {}, "truck": {}
    }

    # Correct class indices according to your model's configuration
    categories = {
        0: "bus",       # Class 0 in your dataset
        1: "car",       # Class 1 in your dataset
        2: "motorbike", # Class 2 in your dataset
        3: "truck"      # Class 3 in your dataset
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        detections_by_category = {key: [] for key in categories.values()}
        for detection in results.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            category = categories.get(class_id)
            if category:
                detections_by_category[category].append([x1, y1, x2, y2])

        for category, detections in detections_by_category.items():
            tracking_data[category] = match_detections_to_tracks(detections, tracking_data[category])

        for category, tracks in tracking_data.items():
            # Assign unique colors for each category
            color = {
                "bus": (0, 0, 255),       # Red
                "car": (0, 255, 0),       # Green
                "motorbike": (255, 0, 0), # Blue
                "truck": (0, 255, 255)    # Yellow
            }.get(category, (255, 255, 255))

            for track_id, track_data in tracks.items():
                bbox = track_data["bbox"]
                center = track_data["center"]
                path = track_data["path"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{category.upper()} ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                for i in range(1, len(path)):
                    cv2.line(frame, path[i - 1], path[i], color, 2)

        display_width = 1200
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_height = int(display_width / aspect_ratio)
        frame = cv2.resize(frame, (display_width, display_height))

        cv2.imshow('YOLO Detection with Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
