import argparse
import io
import cv2
import pytube
from PIL import Image, ImageDraw
import numpy as np
import datetime

import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"


@app.route(DETECTION_URL, methods=["POST"])
def predict_input():
    if not request.method == "POST":
        return jsonify({"error": "Invalid request method"}), 400

    input_type = request.form.get("input_type")
    if input_type == "image":
        return process_image(request)
    elif input_type == "video_url":
        return process_video_url(request)
    elif input_type == "video_path":
        return process_local_video(request)
    else:
        return jsonify({"error": "Invalid input_type"}), 400


def process_image(request):
    if not request.files.get("image"):
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = model(img, size=640)
    detected_objects_list = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({"detected_objects": detected_objects_list})


def process_video_url(request):
    if not request.form.get("video_url"):
        return jsonify({"error": "No video_url provided"}), 400

    video_url = request.form["video_url"]

    # Create a YouTube object to retrieve video details
    yt = pytube.YouTube(video_url)

    # Extract video details
    video_title = yt.title
    video_description = yt.description if yt.description else ""
    channel_name = yt.author

    # Get the highest resolution video stream
    stream = yt.streams.get_highest_resolution()

    # Open the video stream using OpenCV
    cap = cv2.VideoCapture(stream.url)

    detected_objects_list = []  # List to store detected objects

    frame_number = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(img, size=640)

        detected_objects = results.pandas().xyxy[0].to_dict(orient="records")

        for obj in detected_objects:
            # Calculate the timestamp using frame number and frame rate
            timestamp = frame_number / frame_rate
            obj["frame_number"] = frame_number
            # Convert timestamp to human-readable format
            timestamp_hms = str(datetime.timedelta(seconds=timestamp))
            obj["timestamp"] = timestamp_hms

        detected_objects_list.extend(detected_objects)
        frame_number += 1

    cap.release()

    response = {
        "video_title": video_title,
        "video_description": video_description,
        "channel_name": channel_name,
        "detected_objects": detected_objects_list
    }

    return jsonify(response)


def process_local_video(request):
    if not request.form.get("video_path"):
        return jsonify({"error": "No video_path provided"}), 400

    video_path = request.form["video_path"]
    return process_local_video_file(video_path)


def process_local_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    detected_objects_list = []  # List to store detected objects

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(img, size=640)

        detected_objects = results.pandas().xyxy[0].to_dict(orient="records")
        detected_objects_list.extend(detected_objects)

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({"detected_objects": detected_objects_list})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask API exposing yolov5 model")
    parser.add_argument("--port", default=4000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s',
                        help='model to run, i.e. --model yolov5x')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', args.model)
    app.run(host="0.0.0.0", port=args.port)
