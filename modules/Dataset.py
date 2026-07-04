import tensorflow as tf
from .Config import img_size, batch_size, seed, number_aug
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load datasets
def load_dataset(train_dir,test_dir, augment=False,deeplearning=False,preprocess_fn=None):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,  # 20% pour validation   
        subset="training",
        seed=seed,
        image_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True
    )   

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        color_mode="rgb",
        shuffle=False
    )
    
    #compter le nombre total d'images originales
    num_original = tf.data.experimental.cardinality(train_ds).numpy()*batch_size
    print(f"Nombre total d'image train_ds originales : {num_original}")

    #compter le nombre total d'images originales
    num_original = tf.data.experimental.cardinality(val_ds).numpy()*batch_size
    print(f"Nombre total d'image validation originales : {num_original}")

    #compter le nombre total d'images originales
    num_original = tf.data.experimental.cardinality(test_ds).numpy()*batch_size
    print(f"Nombre total d'image test originales : {num_original}")

    AUTOTUNE = tf.data.AUTOTUNE

    # Normalisation 
    if deeplearning:
         train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y))
         val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y))
         test_ds = test_ds.map(lambda x, y: (preprocess_fn(x), y))
    # Data augmentation for training set
    if augment:
        data_augmentation = get_augmentation()
        augmented_ds = train_ds.map(lambda x, y: (tf.cast(data_augmentation(x, training=True),tf.float32), y),
                                num_parallel_calls=AUTOTUNE).repeat(number_aug)
        #appliquer l'augmentation et creer un nouveau dataset
        train_ds = augmented_ds.concatenate(train_ds)
        #compter le total apres augmentation
        num_augmented = tf.data.experimental.cardinality(train_ds).numpy()*batch_size
        print(f"Nombre total d'image apres augmentation : {num_augmented}")
    # Pré-chargement des données pour de meilleures performances
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds