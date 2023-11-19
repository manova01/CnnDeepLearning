# Convolutional Neural Networks (CNN) Deep Learning model
we'll build a model for predicting if we have an image of a bee or a wasp. For this, we will use the "Bee or Wasp?". You can download the dataset for this homework from here:wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/bee-wasp-data/data.zip unzip data.zip.
I  will train a much smaller model from scratch.
## Data Preparation
The dataset contains around 2500 images of bees and around 2100 images of wasps.
The dataset contains separate folders for training and test sets.
Model
For this homework we will use Convolutional Neural Network (CNN). I'll use Keras and model structure will be .
The shape for input should be (150, 150, 3)
Next, created a convolutional layer (Conv2D):
Used 32 filters
Kernel size is (3, 3) (that's the size of the filter)
Used 'relu' as activation
Reduced the size of the feature map with max pooling (MaxPooling2D)
Set the pooling size to (2, 2)
Turned the multi-dimensional result into vectors using a Flatten layer
Next, added a Dense layer with 64 neurons and 'relu' activation
Finally, created the Dense layer with 1 neuron - this will be the output
The output layer would have an activation - used the appropriate activation for the binary classification case
As optimizer used SGD with the following parameters:SGD(lr=0.002, momentum=0.8).
Since we have a binary classification problem, the best loss function for us is binary crossentropy.
## Generators and Trainings
used the following data generator for both train and test sets:
ImageDataGenerator(rescale=1./255)
I  don't need to do any additional pre-processing for the images.
When reading the data from train/test directories, checked the class_mode parameter. Which value should it be for a binary classification problem
Used batch_size=20
Used shuffle=True for both training and test sets.
For training I used .fit() with the following params:
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
The median of training accuracy for all the epochs for this model is 0.769.
The standard deviation of training loss for all the epochs for this model is 0.096.
Data Augmentation
I generated more data using data augmentations.
Added the following augmentations to the training data generator:
rotation_range=50,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest.
Trained the model for 10 more epochs using the same code as previously.
The mean of test loss for all the epochs for the model trained with augmentations is 0.490.
The average of test accuracy for the last 5 epochs (from 6 to 10) for the model trained with augmentations is 0.796.
