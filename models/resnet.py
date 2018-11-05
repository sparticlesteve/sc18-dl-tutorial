"""
ResNet models for Keras.

This currently uses the ResNet implementation from
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py
"""

# Externals
import keras
from keras_contrib.applications import resnet

def build_resnet18_cifar(input_shape=(32, 32, 3), n_classes=10, dropout=None):
    """Build the resnet18 model with appropriate settings for CIFAR10"""

    # These are the recommended settings for CIFAR10 from
    # keras_contrib/applications/resnet.py
    return resnet.ResNet(input_shape=input_shape,
                         classes=n_classes,
                         block='basic',
                         repetitions=[2, 2, 2, 2],
                         include_top=True,
                         dropout=dropout,
                         initial_strides=(1, 1),
                         initial_kernel_size=(3, 3),
                         initial_pooling=None,
                         top='classification')

def build_resnet50(input_shape=(224, 224, 3), n_classes=100, dropout=None,
                   l2_regularization=5e-4):
    """Build the resnet50 model with appropriate settings for ImageNet"""
    model = resnet.ResNet(input_shape=input_shape,
                          classes=n_classes,
                          block='bottleneck',
                          repetitions=[3, 4, 6, 3],
                          dropout=dropout)

    # Increase the L2 regularization to reduce overfitting, as done in
    # https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(l2_regularization)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
    model = keras.models.Model.from_config(model_config)
    return model

def _test():
    model = build_resnet18_cifar()
    model.summary()
