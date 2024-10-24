from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pathlib
import torch
import cv2
import os

app = Flask(__name__)
app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Workaround for PosixPath issue on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLO model
model_path = r'C:\Users\HP\OneDrive\Desktop\Final-Front-end\finals\yolov5\weights\best1.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.eval()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/index')
def serve_index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_path = process_video(filepath)
        return redirect(url_for('display_video', filename=os.path.basename(output_path)))

@app.route('/output/<filename>')
def display_video(filename):
    return render_template('video.html', video_file=filename)

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def process_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer for output (use H.264 codec for browser compatibility)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec for better compatibility
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    # Process video frames
    frame_count = 0
    max_frames = 1000  # Adjust as needed
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if ret:
            # Perform YOLO detection on frame
            results = model(frame)

            # Get annotated frame
            annotated_frame = results.render()[0]

            # Write to output video
            out.write(annotated_frame)
            frame_count += 1
        else:
            break

    # Release resources
    cap.release()
    out.release()

    return output_path

if __name__ == '__main__':
    app.run(debug=True)

pathlib.PosixPath = temp