import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje MNIST skupa podataka
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizacija podataka (pretvaranje u raspon [0,1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Preoblikovanje podataka kako bi odgovarali formatu za CNN (N, 28, 28, 1)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Izgradnja modela
model = models.Sequential([
    # Konvolucijski sloj
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Drugi konvolucijski sloj
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),

    # Treći konvolucijski sloj
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Potpuno povezani slojevi
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompajliranje modela
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dodavanje callback-a za TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

# Dodavanje callback-a za pohranu najboljeg modela
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

# Treniranje modela
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1,  # Koristimo 10% za validaciju
    callbacks=[tensorboard_callback, checkpoint_callback]
)

# Učitavanje najboljeg modela
best_model = tf.keras.models.load_model('best_model.h5')

# Točnost na skupu za učenje
train_loss, train_acc = best_model.evaluate(x_train, y_train)
print(f'Točnost na skupu za učenje: {train_acc}')

# Točnost na skupu za testiranje
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f'Točnost na skupu za testiranje: {test_acc}')

# Predikcije na skupu za učenje i testiranje
y_train_pred = best_model.predict(x_train)
y_train_pred_classes = y_train_pred.argmax(axis=-1)

y_test_pred = best_model.predict(x_test)
y_test_pred_classes = y_test_pred.argmax(axis=-1)

# Matrica zabune za skup za učenje
train_conf_matrix = confusion_matrix(y_train, y_train_pred_classes)

# Matrica zabune za skup za testiranje
test_conf_matrix = confusion_matrix(y_test, y_test_pred_classes)

# Prikazivanje matrice zabune
def plot_confusion_matrix(cm, title="Matrica zabune"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Prediktivna Klasa')
    plt.ylabel('Stvarna Klasa')
    plt.title(title)
    plt.show()

# Prikazivanje matrica zabune
plot_confusion_matrix(train_conf_matrix, "Matrica zabune na skupu za učenje")
plot_confusion_matrix(test_conf_matrix, "Matrica zabune na skupu za testiranje")
