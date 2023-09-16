##### 18-class Classification Model
##### By Aliye Hashemi


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import time
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_auc_score



# record start time
start_time = time.time()


# Load data from csv files
data = pd.read_csv("network_input.csv", header=None)
labels = pd.read_csv("input_labels.txt", header=None)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

# Reshape data for convolutional neural network
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=18)
y_test = keras.utils.to_categorical(y_test, num_classes=18)

# Build model architecture
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=18, activation='softmax'))
model.summary()


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Evaluate model on testing set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

# Predict probabilities for each class
y_pred_prob = model.predict(X_test)

# Calculate AUC for each class
auc_scores = []
target_names = ['Class {}'.format(i) for i in range(18)]
for i in range(len(target_names)):
    auc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
    auc_scores.append(auc)
    print('AUC for', target_names[i], ':', auc)

# Average AUC across all classes
avg_auc = sum(auc_scores) / len(auc_scores)
print('Average AUC:', avg_auc)

# Plot accuracy and loss over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.plot(history.history['accuracy'], label='Training accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Training loss')
ax2.plot(history.history['val_loss'], label='Validation loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.show()
# Generate confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(18)
plt.xticks(tick_marks, range(18))
plt.yticks(tick_marks, range(18))
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.show()

# Save model
model.save("my_model.h5")

# Generate classification report
target_names = ['Class {}'.format(i) for i in range(18)]
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names))

# print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names))
# print('AUC:', auc)

# record end time
end_time = time.time()

# calculate total run time
total_time = end_time - start_time
print("Total run time: {:.2f} seconds".format(total_time))







