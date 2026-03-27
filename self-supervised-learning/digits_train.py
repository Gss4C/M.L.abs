import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/ 255.
x_test  = x_test.astype('float32')/255.

x_train = np.expand_dims(x_train, -1) 
x_test  = np.expand_dims(x_test, -1)


x_train_small = x_train[:10000]
y_train_small = y_train[:10000]
x_test_small = x_test[:2000]
y_test_small = y_test[:2000]

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
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train_small, y_train_small, 
    epochs          = 5, 
    batch_size      = 64, 
    validation_data = (x_test_small, y_test_small)
)

predictions = model.predict(x_test_small)
num_examples = 5
indices = np.random.choice(len(x_test_small), num_examples, replace=False)

for i, idx in enumerate(indices):
    img = x_test_small[idx].squeeze()
    true_label = y_test_small[idx]
    pred_label = np.argmax(predictions[idx])

    plt.subplot(1, num_examples, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

#plt.show() notebooks only
plt.savefig('digits_result.png')
