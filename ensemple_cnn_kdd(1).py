import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('NSL-KDD/NSL_KDD_Train.csv', header=None)


# ..................................cleaning the dataset....................................
# Drop rows with missing values
df.dropna(inplace=True)
# Remove duplicates
df.drop_duplicates(inplace=True)
# Replace infinite values with 10^10
df.replace([np.inf, -np.inf], 1e10, inplace=True)
# # Replace negative values with 0
# df[df < 0] = 0



# Assuming the last column is the target label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode categorical variables in columns 1, 2, and 3
label_encoders = []
for i in range(1, 4):
    label_encoder = LabelEncoder()
    X[:, i] = label_encoder.fit_transform(X[:, i])
    label_encoders.append(label_encoder)

# Normalize numerical features (assuming the remaining columns are numerical)
scaler = StandardScaler()
X[:, 4:] = scaler.fit_transform(X[:, 4:])



# One-hot encode the target labels
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([

    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Conv1D(64, kernel_size=3),
    Dropout(0.2),

    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),

    Bidirectional(LSTM(32)),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Using softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape input data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert input features to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
 
# Train the model
history = model.fit(X_train, y_train, epochs=12, batch_size=64, validation_data=(X_test, y_test))
 
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Predict probabilities for each class
y_probs = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_probs, axis=1)

# Decode one-hot encoded labels
y_true = np.argmax(y_test, axis=1)


# Convert class indices to labels
class_labels = label_binarizer.classes_

# Calculate metrics
print("Confusion Matrix:")

cm = confusion_matrix(y_true, y_pred)
print(cm)

## Get Class Labels
labels = class_labels
class_names = labels
print("************")
print("**********************************************************************Class Labels:",len(class_names))
print("************")
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





# Convert confusion matrix to DataFrame
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true, y_pred))

# Save DataFrame to CSV file
confusion_matrix_df.to_csv('confusion_matrix_cnn_kdd.csv', index=False)

print("\nClassification Report:")
print(classification_report(y_true,  y_pred, zero_division=1)) 


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy - KDD')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

