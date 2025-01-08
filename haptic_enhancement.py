import cv2
import pandas as pd
import supervision as sv
import os
import json
import socket
import paho.mqtt.client as mqtt
import numpy as np
from inference import get_model
from collections import defaultdict, deque
from time import sleep

# Setting up the environment and API linkage
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ROBOFLOW_API_KEY = ""
model = get_model(model_id="vehicle-speed-estimation/2", api_key=ROBOFLOW_API_KEY)

# Define road dimensions in meters
road_width = 7
road_height = 60

# Points for perspective transformation
SOURCE = np.array([[660, 265], [895, 265], [850, 450], [225, 450]])
TARGET = np.array([[0, 0], [road_width-1, 0], [road_width-1, road_height-1], [0, road_height-1]])


# Perspective transformation class
class ViewTransformer:
    """
    Handles perspective transformation of points from one view to another.

    Arguments:
        source (np.ndarray): Source points for transformation.
        target (np.ndarray): Target points for transformation.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms points based on the perspective matrix.

        Arguments:
            points (np.ndarray): Points to transform.

        Returns:
            np.ndarray: Transformed points.
        """
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# Initialize vehicle detection and tracking components
video_path = 'highway.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_info = sv.VideoInfo.from_video_path(video_path)
byte_track = sv.ByteTrack(frame_rate=video_info.fps)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=text_scale, text_thickness=thickness)
polygon_zone = sv.PolygonZone(SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Prepare directories and data structures
if not os.path.exists('detected_frames_sv'):
    os.makedirs('detected_frames_sv')

px = pd.DataFrame()
speeds = {}
test_speed = []
tests = {}
classes = {}
near_ids = {}
id_sent = set()


# MQTT setup functions
def get_local_ip():
    """
    Retrieves the local IP address of the machine.

    Returns:
        str: Local IP address.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))  # Connect to a dummy IP address
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'  # Fallback to localhost
    finally:
        s.close()
    return local_ip


def on_connect(client, userdata, flags, rc):
    """
    Callback function when MQTT client connects to the broker.

    Arguments:
        client (mqtt.Client): MQTT client instance.
        userdata: User data.
        flags: Response flags.
        rc (int): Connection result code.
    """
    if rc == 0:
        print("Connected successfully to broker")
    else:
        print("Connect failed with result code " + str(rc))


broker_address = get_local_ip()
topic = "esp32/output"

# Process video frames and perform vehicle detection
frame_generator = sv.get_video_frames_generator(video_path)
out = cv2.VideoWriter('output_yolo.avi', fourcc, 20.0, video_info.fps)

count = 0
for frame in frame_generator:
    d = 50  # Distance calculation initialization
    count += 1
    print(f"Processing frame: {count}")
    result = model.infer(frame)[0]  # Run inference
    detections = sv.Detections.from_inference(result)
    detections = detections[polygon_zone.trigger(detections)]  # Filter detections by zone
    detections = byte_track.update_with_detections(detections=detections)  # Track detections

    if len(detections.data) == 0:
        continue

    # Transform points and calculate speeds
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points=points).astype(int)
    labels = []

    for tracker_id, [_, y], c in zip(detections.tracker_id, points, detections.data["class_name"]):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"{tracker_id}")
        else:
            coordinates_start = coordinates[tracker_id][-1]
            coordinates_end = coordinates[tracker_id][0]
            distance = abs(coordinates_start - coordinates_end)
            time = len(coordinates[tracker_id]) // video_info.fps
            speed = distance / time * 3.6 if time > 0.3 else 0
            speed = round(speed, 4)
            d -= (speed / 3.6) * time
            labels.append(f"#{tracker_id} {speed}")
            speeds[tracker_id] = float(speed)
            classes[tracker_id] = c

    # Data preparation and MQTT publishing
    if speeds:
        data = {
            'tracker_Id': list(speeds.keys()),
            'speeds': list(speeds.values()),
            'classes': list(classes.values()),
            'near_Ids': list(near_ids.values())
        }

        
        res_ = pd.DataFrame(data)
        id_set = set(res_['tracker_Id'])
        res_["same_line"] = res_['near_Ids'].apply(lambda x: [item for item in x if item in id_set])
        res_['max_speed'] = res_.apply(lambda row: max([speeds[item] for item in row['same_line']], default=row['speeds']), axis=1)
        res_["time_taken"] = d / (res_["max_speed"] / 3.6)
        res_["time_warn"] = res_["time_taken"] - 2

        client = mqtt.Client(client_id="espClient", clean_session=True, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
        client.on_connect = on_connect
        client.connect(broker_address, 1883, 60)
        client.loop_start()

        try:
            s = res_["max_speed"].iloc[-1]
            t = res_["time_warn"].iloc[-1]
            c = res_["classes"].iloc[-1]
            data_ = {'classes': c, "max_speed": s, "time_warn": t}
            client.publish(topic, json.dumps(data_))
            print(f"Sent data: {data_}")
            sleep(1)
        finally:
            client.loop_stop()
            client.disconnect()

    # Annotate and save the frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.BLUE)
    frame_filename = f'detected_frames_yolo/frame_{count}.jpg'
    cv2.imwrite(frame_filename, annotated_frame)
    out.write(annotated_frame)
