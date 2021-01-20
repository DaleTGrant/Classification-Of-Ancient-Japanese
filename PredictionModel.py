import numpy as np
import tensorflow as tf

filepath = './Weights/Originalweights.hdf5'

# Load model at filepath via saved model weights
def LoadModel():
    predictionModel = tf.keras.models.load_model(filepath)
    predictionModel.compile(
    optimizer= "adam",
    loss='categorical_crossentropy',
    metrics=['acc']
    )
    return predictionModel

# Format supplied image to appropriate dimensions and rescale pixels for prediction
def FormatImage(imageToFormat):
    if(len(imageToFormat.shape)==2):
        imageToFormat = np.expand_dims(imageToFormat, axis=-1)
    imageToFormat= imageToFormat/255
    if(imageToFormat.shape[-1]==1):
        imageToFormat = np.concatenate([imageToFormat] * 3,axis=-1)
    return imageToFormat

# Method called by Classify button, model predicts class from model, and returns class and confidence to GUI
def Predict(predictImage):
    predictionModel = LoadModel()
    predictImage = FormatImage(predictImage)
    x = predictionModel.predict(np.expand_dims(predictImage,axis=0))
    predicted_class = np.argmax(x);
    class_confidence = x[0,predicted_class]
    return predicted_class, class_confidence
