import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

# Učitavanje skupa podataka (npr. CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizacija
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Pretvaranje oznaka u one-hot
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Arhitektura CNN-a
model = models.Sequential([
layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Dropout(0.25),

layers.Conv2D(128, (3, 3), activation='relu'),
layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Dropout(0.25),

layers.Flatten(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax')
])

# Provjera broja parametara
model.summary()

# Kompilacija modela
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Callback - čuvanje najboljeg modela
checkpoint_cb = ModelCheckpoint('najbolji_model.h5', save_best_only=True)

# Callback - TensorBoard
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Treniranje modela s 20% validacijskih podataka
history = model.fit(
x_train, y_train_cat,
epochs=20,
batch_size=64,
validation_split=0.2,
callbacks=[checkpoint_cb, tensorboard_cb]
)

# Učitavanje najboljeg modela
model = tf.keras.models.load_model('najbolji_model.h5')

# Evaluacija na testnom skupu
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Točnost na testnim podacima: {test_acc:.4f}")

# Predikcija i matrica zabune
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrica zabune")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.show()
