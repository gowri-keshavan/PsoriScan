import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


#model = load_model(r"C:/Users/Niharika PM/OneDrive/Desktop/cv project/Psoriasis-Detection/psoriasis_model.h5")

model=load_model("psoriasis_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0 
    return img_tensor

def predict_image(img_path):
    img_tensor = preprocess_image(img_path)
    prediction = model.predict(img_tensor)
    label = "Psoriasis" if prediction[0] > 0.5 else "Non-Psoriasis"
    confidence = round(prediction[0][0] * 100, 2)  
    return label, confidence

def highlight_psoriasis(img_path, output_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))  
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2 
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    highlighted = cv2.addWeighted(img_resized, 0.7, heatmap, 0.5, 0)
        cv2.imwrite(output_path, highlighted)
