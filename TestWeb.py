import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def process_yolov8_output(output, confidence_threshold=0.5):
    """
    Process YOLOv8 model output (1,5,8400) format
    Returns list of detections with coordinates and confidence
    """
    detections = []
    output = output[0].T  # Transpose to (8400,5)
    
    for pred in output:
        confidence = pred[4]
        
        if confidence > confidence_threshold:
            x_center, y_center, w, h = pred[:4]
            
            # Convert from center coordinates to corner coordinates
            xmin = x_center - w/2
            ymin = y_center - h/2
            xmax = x_center + w/2
            ymax = y_center + h/2
            
            detections.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'confidence': confidence
            })
    
    return detections

def initialize_model(model_path):
    """Initialize TFLite model and return interpreter"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_objects(interpreter, frame, confidence_threshold=0.5):
    """Run object detection on a single frame"""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Get input shape and resize frame
    input_shape = input_details['shape']
    img_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details['index'], img_batch)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    
    # Process outputs
    return process_yolov8_output(output_data, confidence_threshold)

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame"""
    height, width = frame.shape[:2]
    
    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        xmin = int(det['xmin'] * width)
        ymin = int(det['ymin'] * height)
        xmax = int(det['xmax'] * width)
        ymax = int(det['ymax'] * height)
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Label with confidence
        label = f"Cone: {det['confidence']:.2f}"
        cv2.putText(frame, label, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def webcam_detection(model_path, confidence_threshold=0.5):
    """Run real-time detection from webcam"""
    # Initialize model
    interpreter = initialize_model(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # For FPS calculation
    prev_time = 0
    fps = 0
    
    print("\nStarting webcam detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        detections = detect_objects(interpreter, frame, confidence_threshold)
        
        # Draw detections
        frame_with_detections = draw_detections(frame.copy(), detections)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame_with_detections, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display model info
        cv2.putText(frame_with_detections, f"Model: {os.path.basename(model_path)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Traffic Cone Detection', frame_with_detections)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Select which model to use
    model_choice = input("Choose model (1 for float32, 2 for float16): ")
    
    if model_choice == '1':
        model_path = 'best_float32.tflite'
    elif model_choice == '2':
        model_path = 'best_float16.tflite'
    else:
        print("Invalid choice, using float32 by default")
        model_path = 'best_float32.tflite'
    
    # Start webcam detection
    webcam_detection(model_path, confidence_threshold=0.5)