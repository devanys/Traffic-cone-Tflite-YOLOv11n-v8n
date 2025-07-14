# test_tflite_model.py
import cv2
import numpy as np
import tensorflow as tf
import time
from tabulate import tabulate
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
        # Each prediction: [x_center, y_center, width, height, confidence]
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

def test_tflite_model(model_path, image_path, confidence_threshold=0.5):
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    original_image = image.copy()
    height, width = original_image.shape[:2]
    
    # Get model details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    
    # Preprocess
    input_shape = input_details['shape']
    img_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Inference
    start_time = time.time()
    interpreter.set_tensor(input_details['index'], img_batch)
    interpreter.invoke()
    end_time = time.time()
    
    # Get outputs
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (end_time - start_time)*1000
    
    # Process outputs for YOLOv8 format
    detections = process_yolov8_output(output_data, confidence_threshold)
    
    # Prepare display data
    display_data = []
    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        xmin = int(det['xmin'] * width)
        ymin = int(det['ymin'] * height)
        xmax = int(det['xmax'] * width)
        ymax = int(det['ymax'] * height)
        
        display_data.append({
            'class': 'Traffic Cone',
            'confidence': f"{det['confidence']:.4f}",
            'coordinates': f"({xmin},{ymin})-({xmax},{ymax})",
            'width': xmax - xmin,
            'height': ymax - ymin
        })
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Label with class and confidence
        label = f"Cone: {det['confidence']:.2f}"
        cv2.putText(original_image, label, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display text output
    print(f"\nMODEL: {model_path}")
    print(f"Input shape: {input_shape}")
    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Raw output shape: {output_data.shape}")
    
    if display_data:
        print("\nDETECTION RESULTS:")
        print(tabulate(display_data, headers="keys", tablefmt="grid"))
    else:
        print("\nNo detections above confidence threshold")
    
    # Display visual results
    cv2.imshow(f'Results: {os.path.basename(model_path)}', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'model': model_path,
        'inference_time': inference_time,
        'detections': display_data,
        'raw_output': output_data
    }

if __name__ == "__main__":
    # Test both models
    image_path = 'Cone.jpg'  # Make sure this image exists
    
    print("Testing Float32 model:")
    output_32 = test_tflite_model('best_float32.tflite', image_path)
    
    print("\nTesting Float16 model:")
    output_16 = test_tflite_model('best_float16.tflite', image_path)