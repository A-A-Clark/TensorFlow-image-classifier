import tensorflow as tf
from tensorflow.keras import layers


def load_datasets(data_dir='data/raw', batch_size=32, img_height=224, img_width=224):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{data_dir}/train',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{data_dir}/valid',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{data_dir}/test',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )

    return train_ds, valid_ds, test_ds


def augment_dataset(dataset):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Map the augmentation onto each batch
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    return augmented_dataset


if __name__ == '__main__':
    # Load the datasets
    train_ds, valid_ds, test_ds = load_datasets()

    # Apply data augmentation to the training dataset
    augmented_train_ds = augment_dataset(train_ds)

    # For example, iterate over one batch from the augmented training set
    for images, labels in augmented_train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
