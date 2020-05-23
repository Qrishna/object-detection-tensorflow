import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = VGG16()
print(model.summary())


for file in os.listdir('data'):
    full_path = 'data/' + file
    print(full_path)
    image = load_img(full_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    y_pred = model.predict(image)
    label = decode_predictions(y_pred, top=1)
    print(label)
    print(" ")
