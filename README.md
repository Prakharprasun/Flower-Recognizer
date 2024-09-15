# Flower-Recognizer using EfficientNetB0
Kaggle Notebook: https://www.kaggle.com/code/prakharprasun/flower-recognizer

We are trying to make a model that can classify flowers using pandas, numpy, tensorflow, opencv and EfficientNetB0, a convolutional neural network (CNN), for multi-class classification of flowers from the TensorFlow Flowers dataset.

The image dataset can be downloaded from:
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

To run this project, you'll need the following libraries:
Python 3.x, tensorflow, numpy, pandas, opencv, sklearn
## Install the required packages using pip:
pip install tensorflow numpy pandas opencv-python scikit-learn
## EfficientNetB0
EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.
EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets with similar accuracy
## Data Acquisition
The image dataset is directly downloaded from:
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
## Processing
Resizing: Images are resized to 224x224 as this is the input size required by EfficientNetB0.

Normalization: All pixel values are normalized to fall within the [0, 1] range for better model performance.

Splitting: The dataset is split into training and test sets using an 90-10 split to validate the generalization performance of the model.

Normalization will help our algorithm to train better. The reason we typically want normalized pixel values is because neural networks rely on gradient calculations. These networks are trying to learn how important or how weighty a certain pixel should be in determining the class of an image. Normalizing the pixel values helps these gradient calculations stay consistent, and not get so large that they slow down or prevent a network from training.

## Model Architecture: EfficientNetB0
We use EfficientNetB0, a highly efficient CNN model, which is pre-trained on ImageNet. The use of transfer learning allows for more effective training on smaller datasets like the flower dataset by transferring knowledge from a larger dataset. The pre-trained network is modified as follows:

The pre-trained EfficientNetB0 serves as the feature extractor (with its top layers removed).

A Global Average Pooling layer is added to condense the output.

A fully connected Dense layer with 128 units and ReLU activation captures the higher-level features.

A Dropout layer is added to prevent overfitting.

A final Dense layer with 5 units (corresponding to the 5 flower classes) and softmax activation completes the model.


Transfer learning is a machine learning technique in which knowledge gained through one task or dataset is used to improve model performance on another related task and/or different dataset.1 In other words, transfer learning uses what has been learned in one setting to improve generalization in another setting.
## Model Compilation
The model is compiled using:

Label Encoding: Label Encoding is a technique that is used to convert categorical columns into numerical ones so that they can be fitted by machine learning models which only take numerical data.

Optimizer: Adam, which dynamically adjusts the learning rate.

Loss Function: Sparse Categorical Cross-Entropy, suitable for integer-labeled multi-class classification.

Evaluation Metric: Accuracy, to monitor the classification performance during training and validation.

Training: Runs for 10 epochs with the default batch size of 32.


Label Encoding is a technique that is used to convert categorical columns into numerical ones so that they can be fitted by machine learning models which only take numerical data.

Optimizers are algorithms or methods that are used to change or tune the attributes of a neural network such as layer weights, learning rate, etc. in order to reduce the loss and in turn improve the model.

Adam(Adaptive Moment Estimation) is an adaptive optimization algorithm that was created specifically for deep neural network training. It can be viewed as a fusion of momentum-based stochastic gradient descent and RMSprop. It scales the learning rate using squared gradients, similar to RMSprop, and leverages momentum by using the gradient’s moving average rather than the gradient itself, similar to SGD with momentum. 

The model is trained using the training set and evaluated on the validation set for 50 epochs using Mean Absolute Error (MAE) as a metric.

The model calculates the loss (or error) by comparing its prediction to the actual target value using a loss function. The loss function quantifies how far the model's prediction is from the target.

An epoch (also known as training cycle) in machine learning is a term used to describe one complete pass through the entire training dataset by the learning algorithm. During an epoch, the machine learning model is exposed to every example in the dataset once, allowing it to learn from the data and adjust its parameters (weights) accordingly. The number of epochs is a hyperparameter that determines the number of times the learning algorithm will work through the entire training dataset.
## Model Training and Evaluation
The model is trained using the training set, and its performance is validated using the test set. A total of 10 epochs are used for training, with the following metrics monitored:

Training Loss and Validation Loss

Training Accuracy and Validation Accuracy

The model's validation accuracy is categorized as follows:

Excellent!: Validation accuracy ≥ 90%,
Great!: 80% ≤ Validation accuracy < 90%,
Good.: 70% ≤ Validation accuracy < 80%,
Fair.: 60% ≤ Validation accuracy < 70%,
Needs Improvement.: Validation accuracy < 60%,
## Code Overview
Class: ImageClassifier

Methods
__init__(self, image_url, img_size=(224, 224), test_size=0.1, random_state=2): Initializes the classifier with the dataset URL, image size, and test set size.

download_and_prepare_data(self): Downloads the flower dataset, loads image paths, and assigns labels based on folder structure.

preprocess_data(self): Resizes images to the defined size, normalizes pixel values, and splits data into training and test sets.

build_model(self): Builds the CNN model using EfficientNetB0 as the base model with additional Dense layers.

train_model(self, epochs=10): Trains the model for the specified number of epochs.

evaluate_model(self): Evaluates model performance on the test set, printing validation accuracy and loss.

categorize_val_acc(self): Categorizes the validation accuracy into performance tiers (Excellent, Great, etc.).

print_model_summary(self): Prints a summary of the model architecture.

run(self): Executes the complete workflow, from data downloading to model evaluation and summary display.

## Data
The dataset consists of flower images, classified into five categories: daisies, dandelions, roses, sunflowers, and tulips. The images are fetched using TensorFlow utilities, and each flower class is assigned a corresponding integer label.
## Transfer Learning with EfficientNetB0
EfficientNetB0 is used as the backbone for feature extraction. By leveraging pre-trained weights, the model can extract meaningful features, reducing the training time and improving the accuracy on the flower dataset.
## Results
Upon running the classifier, results such as validation accuracy, model summary, and categorized accuracy feedback are printed. These insights help identify how well the model has learned and generalized to unseen data.
## References
Towardsdatascience

Github

Arxiv

ResearchGate

Medium

IBM

GeeksforGeeks
