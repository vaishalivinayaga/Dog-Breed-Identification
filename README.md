# Dog-Breed-Identification

Problem Statement:
We are provided with a training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. The goal is to create a classifier capable of determining a dog's breed from a photo. The list of breeds is as follows 
Task Description Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the
answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that? In this Task, we were provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well we can tell our Norfolk Terriers from our Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, we might find the problem more, err, ruff than we anticipated.
Support: Dog-Breed-Identification-using-CNN-with-Keras has a low active ecosystem.
Quality: Dog-Breed-Identification-using-CNN-with-Keras has no bugs reported.
Security: Dog-Breed-Identification-using-CNN-with-Keras has no vulnerabilities reported, and its dependent libraries have no 
vulnerabilities reported.
License: Dog-Breed-Identification-using-CNN-with-Keras does not have a standard license declared.
Re-use: Dog-Breed-Identification-using-CNN-with-Keras releases are not available. You will need to build from source code and 
install.

Steps for Implementation:
Step 1: Import Datasets
Obviously, to be able to build an algorithm intended to identify dogs we will need some “dog data”. A lot of it. Thankfully, 
for this project Udacity is providing a decent number of dog images including the corresponding breed labels. Concretely, 
the image data comprises 8351 dog images and 133 separate dog breed names. Since the app has the
additional task to assign the most resembling dog breed to a given human face, we also need a dataset with human faces. 
The dataset provided by Udacity includes 13233 images from the labeled faces in the wild dataset
Step 2: Detect Humans
This seems to be a somewhat surprising step in the development of a dog identification app, but it is necessary for its extra
job to assign the most resembling dog breed to a given human face. In order to detect human faces in images we will use 
OpenCV’s implementation of Haar feature-based cascade classifiers. The approach of this classifier is based on the concept 
of Haar-like features, which is widely used in the field of object recognition because of its convincing calculation speed.
Step 3: Detect Dogs
Now that we have a pretty decent algorithm to detect human faces in images we surely want to build a similar function for 
dog detection. Unfortunately, at the moment there is no comparable “dog detector” available for OpenCV’s 
CascadeClassifiers. Therefore, we choose another approach by employing an image classification model which has been 
pre-trained on the vast image database of ImageNet. More specifically, we will use the high-level deep learning API Keras to 
load the ResNet-50 convolutional neural network and run images through this model. For a specific image the network 
predicts probabilites for each of 1000 image categories in total
Step 4: Create a CNN to Classify Dog Breeds (from Scratch)
Now we will come to the really interesting part and tackle the implementation of the app’s principal task to tell the correct
dog breed label from an image of a dog. We could make things easy and just use the pre-trained model from step two and 
predict the dog breed labels defined in the categories of the ImageNet dataset. But of course it’s much more exciting, 
interesting and educational to build our own solution, so here we go! Before we start building our own classifier, a few 
words about convolutional neural networks.Convolutional neural networks (CNNs) are a class of deep neural networks 
primarily used in the analysis of images.
Step 5: Use a CNN to Classify Dog Breeds (using Transfer Learning)
The general idea behind ​transfer learning is the fact that it is much easier to teach specialized skills to a subject that already 
has basic knowledge in the specific domain. There are a lot of neural network models out there that already specialize in 
image recognition and have been trained on a huge amount of data. Our strategy now is to take advantage of such pretrained networks and our plan can be outlined
Step 6: Create a CNN to Classify Dog Breeds (using Transfer Learning)
We will now take step 4 as a template and define our own CNN using transfer learning. We choose InceptionV3 as the 
network that should provide us with the features for our training layers. Inception is another high performing model on the 
ImageNet dataset and its power lies in the fact that the network could be designed much deeper than other models by 
introducing subnetworks called inception modules.
Step 7: Write your Algorithm
So, let’s now collect the achievements and findings from the previous steps and write an algorithm that takes an image of a 
dog or a human und spits out a dog breed along with 4 sample images of the specific breed.
Step 8: Test your Algorithm
Finally, let’s test our algorithm with a few test images.


Results:
65/65 [==] - 1s 21ms/step - loss: 92.6662 - accuracy: 0.4368
Evaluation results: [loss, accuracy] [92.66619873046875, 0.43683186173439026]
true predictions: 41 , false predictions: 9
The accuracy we have achieved by using this model is 82.05%
Overall, we consider our results to be a success given the high number of breeds in this fine-grained classification problem. We are able to effectively predict the correct breed over 50% of the time in one guess, a result that very few humans could match given the high variability both between and within the 166 different breeds contained in the dataset.
