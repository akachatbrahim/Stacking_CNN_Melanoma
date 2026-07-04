import tensorflow as tf
from modules.Config import input_shape, num_classes
from tensorflow.keras import models, layers

# ============================================================================
# AVAILABLE DL MODELS CONFIGURATION (3 ARCHITECTURES)
# ============================================================================
model_configs = [
     ("EfficientNetV2B0",
     tf.keras.applications.EfficientNetV2B0,
     tf.keras.applications.efficientnet_v2.preprocess_input),

    ("MobileNetV2",
     tf.keras.applications.MobileNetV2,
     tf.keras.applications.mobilenet_v2.preprocess_input),

    ("DenseNet169",
     tf.keras.applications.DenseNet169,
     tf.keras.applications.densenet.preprocess_input)

]