Yes, it's absolutely possible to predict using your own categories list. To do so, you have to train the ResNet model (or any other model you choose) with your own dataset that contains the images of the categories you're interested in. The process generally involves the following steps:

1. **Prepare the Dataset**: You'll need a labeled dataset with images for each category you want to classify. Organize your images in a folder structure where each subfolder represents a category and contains the images belonging to that category.

2. **Preprocess the Images**: Before training, you need to preprocess the images so that they are suitable for the model. This might include resizing them to the required input size of the model (for ResNet50, it's 224x224), normalizing pixel values, and possibly data augmentation to increase the diversity of your training data.

3. **Build the Model**: Use ResNet or another architecture as a base model and adapt it for your specific number of categories by customizing the final layer(s).

4. **Train the Model**: Train the modified model with your dataset. During this step, the model will learn to distinguish between the categories based on the features extracted from the images.

5. **Evaluate the Model**: Validate the model performance using a test set that the model hasn't seen during training to ensure that it generalizes well.

6. **Predictions**: Once the model is trained, you can use it to predict the categories of new images using your custom category list.

Here is a simplified code outline to demonstrate these steps using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directory where your training and validation data is stored
training_data_dir = 'path_to_training_data'
validation_data_dir = 'path_to_validation_data'

# Create ImageDataGenerators for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    training_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Use 'binary' for binary classification
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Use 'binary' for binary classification
)

# Download the ResNet50 model with weights pre-trained on ImageNet, without the top layer (classifier)
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze the layers of the base model to not train them during the first training pass
for layer in base_model.layers:
    layer.trainable = False

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer — we have n categories
predictions = Dense(n, activation='softmax')(x)

# The model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a loss function, an optimizer, and metrics to watch
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the new data for a few epochs
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // 32
)

# Predict new images
# ... Load and preprocess your image as shown before ...
preds = model.predict(x)

# ... Your code to handle the predictions, e.g., show the predicted category ...
```

Please adapt the paths, number of categories (`n` in the dense layer), and other parameters as needed for your particular dataset and problem. After training, you'll be able to use your model to predict the categories that you've defined based on the images it has been trained on.