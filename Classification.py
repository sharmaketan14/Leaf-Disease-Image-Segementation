import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Sequential
import tensorflow as tf
from flask import *
import os
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
import cv2
from sklearn.cluster import KMeans
import time

app = Flask(__name__)

def input_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    return image

def pre_processing_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    _ , mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.equalizeHist(mask)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    #mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    
    #cv2.imwrite('thresholding.png', cv2.hconcat([image, np.stack((mask, mask, mask), axis=2)]))
    
    return mask

def get_mean_color_df(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    contours_img_before_filtering = mask.copy()
    contours_img_before_filtering = cv2.cvtColor(contours_img_before_filtering, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_img_before_filtering, contours, -1, (0, 255, 0), 3)
    
    
    filtered_contours = []
    df_mean_color = pd.DataFrame()
    for idx, contour in enumerate(contours):
        area = int(cv2.contourArea(contour))

        # if area is higher than 3000:
        if area > 100:
            filtered_contours.append(contour)
            # get mean color of contour:
            masked = np.zeros_like(image[:, :, 0])  # This mask is used to get the mean color of the specific bead (contour), for kmeans
            cv2.drawContours(masked, [contour], 0, 255, -1)

            B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
            df = pd.DataFrame({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean}, index=[idx])
            df_mean_color = pd.concat([df_mean_color, df])

    df_mean_color.head()
    return contours, df_mean_color

def k_means_clustering(df_mean_color):
    km = KMeans( n_clusters=2, n_init=2)
    df_mean_color['label'] = km.fit_predict(df_mean_color)

    return df_mean_color

def draw_segmented_objects(image, contours, label_cnt_idx, bubbles_count):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image = cv2.putText(masked_image, f'{bubbles_count} bubbles', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 3, color = (255, 255, 255), thickness = 10, lineType = cv2.LINE_AA)
    return masked_image

def get_segmented_image(df_mean_color, contours, image):
    img = image.copy()
    for label, df_grouped in df_mean_color.groupby('label'):
        bubbles_amount = len(df_grouped)
        masked_image = draw_segmented_objects(image, contours, df_grouped.index, bubbles_amount)
        img = cv2.hconcat([masked_image])
    return img

def save_segmentation(path):
    image = input_image(path)
    mask = pre_processing_image(image)
    contours, df = get_mean_color_df(mask, image)
    df = k_means_clustering(df)
    img = get_segmented_image(df, contours, image)

    cv2.imwrite('./static/color_segmentation.png', img)


def predict_disease(path, model):
    classes_of_diseases = {'Nematodes':0, "Normal":1, "Bacteria":2, "Fungi":3, "Virus":4}
    img = Image.open(path, 'r').convert("RGB")
    img = img.resize((200, 200))
    w, h = img.size
    image = list(img.getdata())
    mat = np.array(image).reshape((w, h, 3))
    mat = mat/255.0
    l = [mat]
    mat = np.array(l, dtype='float32')
    ans = model.predict(mat)
    
    key_list = list(classes_of_diseases.keys())
    val_list = list(classes_of_diseases.values())
    
    for i in range(0, 5):
        if(ans[0][i] == 1):
            position = val_list.index(i)
            return key_list[position]
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        
        save_segmentation(file_path)
        
        new_model = tf.keras.models.load_model('./prediction_model')
        
        time.sleep(3)
        
        result = predict_disease(file_path, new_model)

        #os.remove(file_path)
        return result
    
if __name__ == '__main__':
    app.run(debug=True)