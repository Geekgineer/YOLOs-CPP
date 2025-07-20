from ultralytics import YOLO
import cv2
import time
import torch 

# Check if CUDA is available (should print False for CPU)
print("CUDA Available:", torch.cuda.is_available())
print("PyTorch Version:", torch.__version__)

# Load the YOLOv11n model and force it to use CPU
model = YOLO("yolo11n.pt").to('cpu')

# Open a video file or use a webcam (use '0' for webcam)
video_path = "/home/elbahnasy/CodingWorkspace/YOLOs-CPP/data/dogs.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Initialize variables to calculate FPS
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference on the current frame
    results = model(frame)
    
    # Render the results on the frame
    frame_with_boxes = results[0].plot()  # this will draw bounding boxes on the frame
    
    # Display the frame
    cv2.imshow("YOLOv11n Inference", frame_with_boxes)
    
    frame_count += 1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print FPS after processing
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time

print(f"Frames per second (FPS): {fps:.2f}, processed {frame_count} frames in {elapsed_time:.2f} seconds.")

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
