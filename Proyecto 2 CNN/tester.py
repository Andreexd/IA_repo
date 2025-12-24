import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize


def num1():
    model = load_model("clasificadoor_animales.h5")

    labels = ["catarina", "gato", "hormiga", "perro", "turtle"]

    target_w = 224
    target_h = 224

    def preprocess_image(img_path):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError("Image not found:", img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        old_h, old_w = img.shape[:2]
        scale = min(target_w / old_w, target_h / old_h)
        new_w = int(old_w * scale)
        new_h = int(old_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        padded = padded.astype("float32") / 255.0

        padded = np.expand_dims(padded, axis=0)

        return padded

    img_path = 'test/HOR2.png'

    X = preprocess_image(img_path)

    prediction = model.predict(X)
    predicted_class = np.argmax(prediction)

    print("Predicted class:", predicted_class)
    print("Label:", labels[predicted_class])

    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title("Es un: " + labels[predicted_class])
    plt.axis("off")
    plt.show()

    modelo_h5 = '|.h5'
    riesgo_model = load_model(modelo_h5)

    images = []
    filenames = ['/home/panque/repos/IA/Eigenface/animals/test/cat3.jpg','/home/panque/repos/IA/Eigenface/animals/test/dog.jpg',
                 '/home/panque/repos/IA/Eigenface/animals/test/cat2.jpg','/home/panque/repos/IA/Eigenface/animals/test/ant.jpg',
                 '/home/panque/repos/IA/Eigenface/animals/test/ladybug.jpg','/home/panque/repos/IA/Eigenface/animals/test/turtle.jpg']

    for filepath in filenames:
        image = plt.imread(filepath)
        images.append(image)

    X = np.array(images, dtype=np.uint8)
    test_X = X.astype('float32')
    test_X = test_X / 255.

    predicted_classes = riesgo_model.predict(test_X)

    sriesgos = ["catarina", "gato", "hormiga", "perro", "turtle"]

    for i, img_tagged in enumerate(predicted_classes):
        print(filenames[i], sriesgos[np.argmax(img_tagged)])

if __name__ == "__main__":
    num1()