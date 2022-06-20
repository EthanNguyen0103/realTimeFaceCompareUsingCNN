import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('face.h5')


def get_className(classNo):
    if classNo == 0:
        return "Hoai Linh"
    elif classNo == 1:
        return "Hong Dang"
    elif classNo == 2:
        return "Hong Van"
    elif classNo == 3:
        return "Hua Vi Van"
    elif classNo == 4:
        return "Lan Phuong"
    elif classNo == 5:
        return "Manh Truong"
    elif classNo == 6:
        return "Ngo Thanh Van"
    elif classNo == 7:
        return "Ninh Duong Lan Ngoc"
    elif classNo == 8:
        return "Tang Thanh Ha"
    elif classNo == 9:
        return "Thai Hoa"
    elif classNo == 10:
        return "Thanh Huong"
    elif classNo == 11:
        return "Thu Quynh"
    elif classNo == 12:
        return "Tran Thanh"
    elif classNo == 13:
        return "Trung Dung"
    elif classNo == 14:
        return "Viet Anh"


while True:
    sucess, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        # turn the image into a numpy array
        image_array = np.asarray(img)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        prediction = model.predict(data)
        # classIndex=model.predict_classes(img)
        classIndex = np.argmax(prediction, axis=-1)
        print("dudoan", classIndex)
        print(prediction)
        probabilityValue = np.amax(prediction)
        if classIndex == 0:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 1:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 2:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 3:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 4:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 5:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 6:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 7:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 8:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 9:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 10:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 11:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 12:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 13:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif classIndex == 14:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)

        cv2.putText(imgOrignal, "Do chinh xac: " + str(round(probabilityValue, 2)*100), (0, 20), font, 0.75,
                    (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('c'):
        k = cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
