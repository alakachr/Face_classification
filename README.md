# Face_classification
This is a binary classification  task for human faces. We train a _mobilenet_v2_ model. Hyperparameters are in the [config file](./src/face_classification/config.yaml)

## Data Analysis

We have an unbalanced class problem. Here is the distribution of classes:
![Class distribution](./assets/unbalanced.png)

When training a model on that data distribution we reach a 100% accuracy for class 1 and a 0% accuracy on class 0.
To solve this, we  drop samples from class1 to balance the distribution : 

![Class distribution](./assets/balanced.png)

After that, we are able to obtain better performance metrics: 

'Class Accuracy': [0.9389, 0.9272],  
'Class Precision': [0.9282, 0.9380],  
'Class Recall': [0.9389, 0.9272],  
'Global Accuracy': 0.9330,  
'Global Precision': 0.9330,  
'Global Recall': 0.9330, 
