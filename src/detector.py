# Object detection logic (loading model, prediction)
from tensorflow.keras.models import load_model
import joblib
from preprocessing import preprocessing
from segmentation import *
from feature_extraction import *
model = load_model('models/animal_detection.h5')
encoder = joblib.load('models/label_encoder.joblib') 

def predict_and_visualize(image, model, encoder, min_area=1200, margin=10):
   
    output_img = image.copy()
    
    processed_img = preprocessing(image)
    
    boxes = extract_bounded(processed_img, min_area, margin)
    
    if not boxes:
        print("Cannot find ROI")
        cv2_imshow(image)
        return []
    
    rois = extract_rois(image, boxes)  
    
    predictions = []
    
    for i, ((x1, y1, x2, y2), roi) in enumerate(zip(boxes, rois)):
        processed_roi = processed_img[y1:y2, x1:x2]
        features = feature_extractor(processed_roi)
        
        prediction = model.predict(np.array([features]))
        predicted_class = encoder.inverse_transform(prediction)[0][0]
        predictions.append(predicted_class)
        
        color = (0, 255, 0)  
        thickness = 2
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(predicted_class, font, font_scale, font_thickness)
        
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        cv2.putText(output_img, predicted_class, 
                   (text_x, text_y),
                   font, font_scale, color, font_thickness)
    
   
    cv2_imshow(output_img)
    
    return predictions