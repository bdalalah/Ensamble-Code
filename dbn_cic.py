import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define RBM class
class RBM(tf.keras.layers.Layer):
    def __init__(self, num_hidden_units, num_visible_units):
        super(RBM, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.num_visible_units = num_visible_units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_visible_units, self.num_hidden_units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='weights')
        self.b_hidden = self.add_weight(shape=(self.num_hidden_units,),
                                        initializer='random_normal',
                                        trainable=True,
                                        name='bias_hidden')
        self.b_visible = self.add_weight(shape=(self.num_visible_units,),
                                         initializer='random_normal',
                                         trainable=True,
                                         name='bias_visible')

    def call(self, inputs):
        # Compute hidden activations
        hidden_activations = tf.sigmoid(tf.matmul(inputs, self.W) + self.b_hidden)
        # Compute visible activations (reconstruction)
        visible_activations = tf.sigmoid(tf.matmul(hidden_activations, tf.transpose(self.W)) + self.b_visible)
        return visible_activations

# Load the dataset
df = pd.read_csv('CIC-DDoS2019/cicddos2019_dataset.csv')

df = df.iloc[:, :-1]   # remove in case of binary *****************


# ..................................cleaning the dataset....................................
# Drop rows with missing values
df.dropna(inplace=True)
# Remove duplicates
df.drop_duplicates(inplace=True)
# Replace infinite values with 10^10
df.replace([np.inf, -np.inf], 1e10, inplace=True)
# # Replace negative values with 0
# df[df < 0] = 0


# the last column is the target label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize numerical features
scaler = StandardScaler()
X[:, :78] = scaler.fit_transform(X[:, :78])

# One-hot encode the target labels
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the ensemble model with RBM features
num_visible_units = X_train.shape[1]
num_hidden_units = 512


rbm_input = tf.keras.Input(shape=(num_visible_units,))
x = RBM(num_hidden_units=num_hidden_units, num_visible_units=num_visible_units)(rbm_input)
rbm_output = RBM(num_hidden_units=num_hidden_units, num_visible_units=num_visible_units)(x)
rbm_model = tf.keras.Model(inputs=rbm_input, outputs=rbm_output)

# Compile the model
rbm_model.compile(optimizer=Adam(), loss='mean_squared_error')  # Use mean squared error for RBM pre-training

# Train the model using RBM features
history = rbm_model.fit(X_train, X_train, epochs=5, batch_size=32, validation_data=(X_test, X_test))

# Extract features using RBM layers
rbm_features_train = rbm_model.predict(X_train)
rbm_features_test = rbm_model.predict(X_test)

# Define the rest of your ensemble model
ensemble_model = Sequential([
    Dense(128, activation='relu', input_shape=(rbm_features_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')

])



# Compile the ensemble model
ensemble_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the ensemble model
ensemble_history = ensemble_model.fit(rbm_features_train, y_train, epochs=12, batch_size=64, validation_data=(rbm_features_test, y_test))

# Evaluate the ensemble model
ensemble_loss, ensemble_accuracy = ensemble_model.evaluate(rbm_features_test, y_test)
print(f'Ensemble Model Test Loss: {ensemble_loss}, Test Accuracy: {ensemble_accuracy}')

# Predict probabilities for each class
y_pred_prob = ensemble_model.predict(rbm_features_test)
# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)
# Convert one-hot encoded labels to class labels
y_true = np.argmax(y_test, axis=1)



# Convert class indices to labels
class_labels = label_binarizer.classes_

# Print confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)


## Get Class Labels
labels = class_labels
class_names = labels

import seaborn as sns




# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()












confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
# Save DataFrame to CSV file
confusion_matrix_df.to_csv('confusion_matrix_dbn_cic.csv', index=False)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

# Draw training and validation accuracy figure

# Plot training and validation accuracy
plt.plot(ensemble_history.history['accuracy'], label='Training Accuracy')
plt.plot(ensemble_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()