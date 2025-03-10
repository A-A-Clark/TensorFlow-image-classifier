import tensorflow as tf
from data_loader import load_datasets, augment_dataset
from model import build_resnet50_model


def main():
    # Load the datasets from the data/raw directory
    train_ds, valid_ds, test_ds = load_datasets()

    # Apply data augmentation to the training dataset
    augmented_train_ds = augment_dataset(train_ds)

    # Build the ResNet50-based model
    model = build_resnet50_model()

    # Compile the model with an optimizer, loss function, and evaluation metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks for checkpointing and TensorBoard logging
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/flower_classifier.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    # Train the model
    history = model.fit(
        augmented_train_ds,
        validation_data=valid_ds,
        epochs=20,
        callbacks=callbacks
    )

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


if __name__ == '__main__':
    main()
