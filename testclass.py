import numpy as np  # linear algebra
import cv2
from keras.models import load_model
from scipy.spatial import distance
from PIL import Image
import base64
from io import BytesIO


model = load_model('./models/low-light-enhancement-with-cnn-version-2.h5')
face_mask_model = load_model('./models/masknet.h5')

modelFile = "./caffemodel/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./caffemodel/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to determine whether is dark or not


def img_estim(img, thrshld):
    is_light = np.mean(img) > thrshld
    return 'light' if is_light else 'dark'

# Function to resize an image to square while keeping its aspect ratio


def resize2SquareKeepingAspectRation(img, size, interpolation):
    print(type(img))
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        return cv2.resize(img, (size, size), interpolation)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w)/2)
    y_pos = int((dif - h)/2)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


mask_label = {0: 'MASK', 1: 'NO MASK'}
dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}

MIN_DISTANCE = 155

InputPath = "./test-images/"

# Selecting an image to test


def identifyFaceMask(image_path):
    # Read image and process for input
    img_to_test = cv2.imread(image_path)
    img_to_test = cv2.cvtColor(img_to_test, cv2.COLOR_BGR2RGB)
    img_to_test = resize2SquareKeepingAspectRation(
        img_to_test, 500, cv2.INTER_AREA)

    if img_estim(img_to_test, 10) == "light":
        light_dark = "light"
        img_for_face_detection = img_to_test
    else:
        light_dark = "dark"
        img_to_test = cv2.cvtColor(img_to_test, cv2.COLOR_BGR2RGB)
        img_to_test = resize2SquareKeepingAspectRation(
            img_to_test, 500, cv2.INTER_AREA)
        img_to_test = img_to_test.reshape(1, 500, 500, 3)
        img_to_test_prediction = model.predict(
            img_to_test)
        img_to_test_prediction = img_to_test_prediction.reshape(
            500, 500, 3)
        img_to_test_prediction = img_to_test_prediction.astype('uint8')
        img_for_face_detection = img_to_test_prediction

    h, w = img_for_face_detection.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        np.float32(img_for_face_detection), (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    enhanced_image_faces = net.forward()
    temp_faces = []

    # to draw faces on image
    for i in range(enhanced_image_faces.shape[2]):
        confidence = enhanced_image_faces[0, 0, i, 2]
        if confidence > 0.5:
            box = enhanced_image_faces[0, 0, i,
                                       3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            temp_faces.append(box.astype("int"))

    # all the faces found in the image
    enhanced_image_faces = temp_faces

    # declare variable for face mask detection
    face_mask_detection_img = img_for_face_detection

    prediction_results = []

    label = [0 for i in range(len(enhanced_image_faces))]
    for i in range(len(enhanced_image_faces)-1):
        for j in range(i+1, len(enhanced_image_faces)):
            dist = distance.euclidean(
                enhanced_image_faces[i][2:], enhanced_image_faces[j][:2])
            secondary_dist = distance.euclidean(
                enhanced_image_faces[i][:2], enhanced_image_faces[j][2:])
            if dist < MIN_DISTANCE or secondary_dist < MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    for i in range(len(enhanced_image_faces)):
        (x, y, w, h) = enhanced_image_faces[i]
        crop = face_mask_detection_img[y:h, x:w]
        crop = resize2SquareKeepingAspectRation(
            crop, 128, cv2.INTER_AREA)
        crop = np.reshape(crop, [1, 128, 128, 3])/255.0
        mask_result = face_mask_model.predict(
            crop)
        prediction_results.append(mask_result)
        # crop.reshape(128,128,3)
        # inidividual faces

        cv2.putText(face_mask_detection_img, mask_label[mask_result.argmax(
        )], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_label[label[i]], 2)
        cv2.rectangle(face_mask_detection_img, (x, y),
                      (w, h), dist_label[label[i]], 2)

    face_mask_detection_img = Image.fromarray(
        face_mask_detection_img, 'RGB')

    return (light_dark, prediction_results, face_mask_detection_img)


light_dark, prediction_results, face_mask_detection_img = identifyFaceMask(
    InputPath + "maksssksksss244.png")
output_image = face_mask_detection_img
