import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np 

def rotate_images(images, angles):
    '''
    Crea le immagini ruotate per ognuno degli angoli che gli indico e gli aggiunge anche la label
    '''
    rotated_images = []
    labels = []
    for img in images:
        for i, angle in enumerate(angles):
            rotated = tf.image.rot90(img, k=angle // 90)
            rotated_images.append(rotated.numpy())
            labels.append(i)
    return np.array(rotated_images), np.array(labels)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/ 255. #normalizzazione pixel da int [0,255] a float [0,1]
x_test  = x_test.astype('float32')/255.
x_train = np.expand_dims(x_train, -1) # Conv-layers expects input like: (N_images, H_pixels, L_pixels, N_channels): qui ho singolo canale (RGB), ma devo aggiungere la colonna che lo tiene
x_test  = np.expand_dims(x_test, -1)

# crete small sub-datasets for ssl
x_train_small = x_train[:10000]
x_test_small = x_test[:2000]

angles = [0,90,180,270]
x_train_rot, y_train_rot = rotate_images(x_train_small, angles)
#x_train_rot, y_train_rot = rotate_images(x_train, angles)
x_test_rot, y_test_rot = rotate_images(x_test_small, angles)
#x_test_rot, y_test_rot = rotate_images(x_test, angles)

##################
# NEURAL NETWORK #
##################

model = models.Sequential([
    layers.Input(shape = (28,28,1)),
    layers.Conv2D(32, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(len(angles), activation='softmax')
])



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_rot, y_train_rot, epochs=5, batch_size=64, validation_data=(x_test_rot, y_test_rot))

import matplotlib.pyplot as plt

predictions = model.predict(x_test_rot)

num_examples = 5
indices = np.random.choice(len(x_test_rot), num_examples, replace=False)

for i, idx in enumerate(indices):
    img = x_test_rot[idx].squeeze()
    true_label = y_test_rot[idx]
    pred_label = np.argmax(predictions[idx])

    plt.subplot(1, num_examples, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {angles[true_label]}°\nPred: {angles[pred_label]}°")
    plt.axis('off')

#plt.show() notebooks only
plt.savefig('result.png')