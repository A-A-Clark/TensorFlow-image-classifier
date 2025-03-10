import tensorflow as tf
from tensorflow.keras import layers, models


def build_resnet50_model(input_shape=(224, 224, 3), num_classes=102, dropout_rate=0.5):
    # Load ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',  # Use pre-trained weights from ImageNet.
        include_top=False,  # Exclude the fully-connected layers.
        input_shape=input_shape
    )

    # Freeze the base model to prevent its weights from being updated during training.
    base_model.trainable = False

    # Define model input.
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess the input images for ResNet50.
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    # Pass the preprocessed input through the ResNet50 base model.
    x = base_model(x, training=False)

    # Apply Global Average Pooling to reduce the spatial dimensions.
    x = layers.GlobalAveragePooling2D()(x)

    # Add a dropout layer for regularisation.
    x = layers.Dropout(dropout_rate)(x)

    # Final dense layer with softmax activation for multi-class classification.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model.
    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # Build and print the model summary.
    model = build_resnet50_model()
    model.summary()
