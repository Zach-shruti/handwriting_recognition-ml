import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance
import pickle  # Importing pickle for saving models

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target.astype(np.int8)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_dt = dt_classifier.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")

# Save the Decision Tree model using pickle
with open('dt_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(dt_classifier, model_file)


# Train a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = rf_classifier.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Save the Random Forest model using pickle
with open('rf_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    
    # Optional: Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Adjust contrast as needed
    
    # Invert colors if the digit is white on black background
    img = np.array(img)
    if np.mean(img) > 128:  # This checks if the image is mostly light (background)
        img = 255 - img  # Invert to ensure digit is black
    
    img = img / 255.0  # Normalize
    img = img.reshape(1, -1)  # Flatten to match the model input shape
    return img

# Example usage with a custom image
image_path = 'image.png'  
custom_image = preprocess_image(image_path)

# Make predictions using the trained models
prediction_dt = dt_classifier.predict(custom_image)
prediction_rf = rf_classifier.predict(custom_image)

print(f"Decision Tree Prediction: {prediction_dt}")
print(f"Random Forest Prediction: {prediction_rf}")
