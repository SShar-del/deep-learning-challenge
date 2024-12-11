# deep-learning-challenge


# Report on the Neural Network Model

## Overview of the analysis

The purpose of this analysis is to train a machine learning model, leveraging deep learning and neural networks, to assist Alphabet Soup in identifying applicants most likely to succeed in their ventures. By analyzing a dataset of over 34,000 organizations that have received funding from Alphabet Soup in the past, the goal is to build a binary classifier capable of predicting the likelihood of success for new applicants seeking funding.


## Results

### Data Preprocessing

- The variable IS_SUCCESSFUL is the target for the model.

- The variables  'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION',’STATUS’, 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', ‘ASK_AMT’ are the features for the model.

- The variables that were removed from the data because they are neither targets nor features are ‘EIN’ and ‘NAME’ as these were merely the identification variables.


### Compiling, Training, and Evaluating the Model

Ref: Base Model Screenshot  

- The neurons selected were based on the number of input features (36), calculated using len(X_train[0]), to ensure the model has sufficient capacity to learn meaningful patterns from the dataset.

- Two hidden layers were chosen to strike a balance between model complexity and computational efficiency, allowing the network to learn more complex relationships without overfitting.

    For the first hidden layer, the number of neurons selected were three times the number of input features (108). This choice provides a robust starting point for learning, capturing a wide range of features and interactions among inputs.

    For the second hidden layer, the number of neurons was reduced to two times the number of input features (72). This helps the model progressively distill and refine the learned representations, focusing on the most critical patterns.

- The “relu” activation function was used for both hidden layers because it is computationally efficient and helps mitigate the vanishing gradient problem, ensuring better convergence during training. The “sigmoid” activation function was used for the output layer because it outputs a probability between 0 and 1, aligning with the binary classification goal of predicting the likelihood of success


## Target model performance

The model was only able to achieve an accuracy around 72% which is below the targeted accuracy of 75%.

## Attempts to increase model performance

Three Optimization attempts were taken to increase the model performance for which the following steps were taken:

### Optimization Attempt 1                                   

Ref: Optimized Model 1 Screenshot  

Adjusting the binning criteria for the catch all bin ("Other") for Application column, created bins for ASK_AMT column. It was done because Binning simplified the  APPLICATION_TYPE and ASK_AMT data, reducing the effect of outliers on the model which could help the model identify patterns associated with specific ranges of funding amounts.Also, by grouping values into broader categories, the model generalizes better and prevents overfitting. 

In the case of my model, the performance slightly reduced to 71%, perhaps because binning introduced a loss of granularity and reduced feature variability. 

### Optimization Attempt 2                            

Ref: Optimized Model 2 Screenshot  

Reverting back to the original APPLICATION_TYPE binning ,  ASK_AMT without binning as done in the starter code since binning reduced performance. 
Added one more hidden layer - Third Hidden layer and reduced epochs from 100 to 75. Adding a third hidden layer can sometimes improve the optimization of a neural network by allowing it to learn more complex patterns and hierarchical representations in the data. Also, more layers introduce additional nonlinearity, which can allow the model to approximate complex functions more effectively.

In the case of my model, the performance increased by a few decimal points but still remained in the overall 72%


### Optimization Attempt 3

Ref: Optimized Model 3 Screenshot  

Adjusting (increase) Neurons in the first hidden layer from 3 to 4 times the number of input features and second hidden layer from 2 to 3 times the number of input features, using different activation functions for the hidden layers, “relu” for the first hidden layer, “leaky_relu” for the second hidden layer. Also using learning rate (.001), batch size (64), and the number of epochs from 100 to 50.

In the case of my model, the performance increased by a few decimal points but still remained in the overall 72%


## Summary

The deep learning model developed for Alphabet Soup achieved an accuracy of approximately 72%, which falls short of the targeted performance of 75%. Several optimization attempts were made to improve the model, including adjustments to data preprocessing, network architecture, and hyperparameters. Despite these efforts, the performance gains were minimal, suggesting that the current approach may have reached its performance ceiling given the dataset and architecture used.

#### Recommendation for a Different Approach:

Given the limited success of optimizing the deep learning model, a different machine learning approach could be more effective for solving this binary classification problem. Specifically, the Ensemble Methods (e.g., Random Forest ) as it excels at handling structured/tabular data and often outperform deep learning models for such tasks.These models can capture non-linear relationships and interactions between features without requiring extensive preprocessing or feature engineering. Outliers and feature scaling also have less impact on tree-based models.




## Acknowledgements

IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links