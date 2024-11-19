from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Define the class names (example labels, you should replace them with actual labels from your dataset)
CLASS_NAMES = ['Healthy', 'Diseased']  # Replace with your actual class labels

# Load the base ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Add custom layers for regression
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='linear')(x)  # Single neuron with linear activation for regression

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Load and preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('path_to_your_dataset', target_size=(256, 256), batch_size=32, subset='training')
val_generator = train_datagen.flow_from_directory('path_to_your_dataset', target_size=(256, 256), batch_size=32, subset='validation')

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the trained model
model.save('plant_disease_regression_model.h5')