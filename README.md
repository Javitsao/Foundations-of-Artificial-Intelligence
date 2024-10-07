# Foundations-of-Artificial-Intelligence


## [Homework 1: Search Algorithms for Pacman](https://www.csie.ntu.edu.tw/~stchen/teaching/fai24/hw1/hw1.html)

### Introduction:
In this assignment, we build general-purpose search algorithms and apply them to solving puzzles in the Pacman game. We implement pathfinding algorithms, including BFS and A*, to solve different tasks.

### Tasks:
- **Part 1:** Finding a single dot
- **Part 2:** Finding all corners
- **Part 3:** Finding multiple dots
- **Part 4:** Fast heuristic search for multiple dots

---

## [Homework 2: Pacman Agents](https://www.csie.ntu.edu.tw/~stchen/teaching/fai24/hw2/hw2.html)

### Introduction:
In this assignment, we design agents for the classic Pacman game, including both Pacman and ghosts. The focus is on implementing minimax, expectimax, and alpha-beta pruning, and creating evaluation functions for better decision-making.

### Tasks:
- **Part 1:** Reflex Agent
- **Part 2:** Minimax
- **Part 3:** Alpha-Beta Pruning
- **Part 4:** Expectimax
- **Part 5:** Evaluation Function

---

## Homework 3: Supervised Machine Learning Pipeline

### Introduction:
In this homework, we build a supervised machine learning pipeline from scratch, including preprocessing, training, and evaluating linear and nonlinear models. We apply the pipeline to both a classification task using the Iris dataset and a regression task using the Boston Housing dataset.

### Tasks:
1. **Dataset:**
   - **Classification task:** [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)
   - **Regression task:** [Boston Housing dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)
   
2. **Preprocessing:**
   - Load and split the dataset into 70% train and 30% test sets.
   - Implement feature scaling: normalize or standardize features.

3. **Models:**
   - Linear model (logistic regression for classification, linear regression for regression).
   - Nonlinear model (decision tree for classification and regression).
   - Random forest model for classification and regression.

4. **Training:**
   - Implement gradient descent for logistic and linear regression.
   - Implement algorithms for building decision trees and random forests.

5. **Evaluation:**
   - Implement accuracy for classification and mean squared error (MSE) for regression.
   - Compare the performance of linear and nonlinear models.

---

## Homework 4: Facial Recognition System

### Introduction:
The goal is to create a simplified facial recognition system using unsupervised learning models. We focus on implementing principal component analysis (PCA) and autoencoders, including denoising autoencoders, to extract and reconstruct features from face images.

### Tasks:
1. **Principal Component Analysis:**
   - Implement the `fit` method to calculate the mean and eigenvectors.
   - Implement the `reconstruct` method and transform the image `subject_05_17.png` using PCA.
   - Implement the `transform` method and use the transformed features to train a logistic regression classifier.

2. **Autoencoder:**
   - Implement the `fit` method to optimize reconstruction error.
   - Implement the `reconstruct` method to transform and reconstruct the image.
   - Implement the `transform` method and use it to train a logistic regression classifier.

3. **Denoising Autoencoder:**
   - Implement the `fit` method to optimize reconstruction with added Gaussian noise.
   - Implement the `reconstruct` method to transform and reconstruct the image.
   - Plot the reconstruction error as a function of iterations or epochs.

---

## [Final Project: Texas Hold’em AI](https://docs.google.com/presentation/d/17Hx5R2BoehE-IvOlOrsKZnYmaiEYz1-_-MAetO-KGdY/edit?usp=sharing)

### Introduction:
For this project, I implemented two different methods to play poker (Texas Hold’em). The first method uses a heuristic strategy based on real-world experience, while the second method uses the **Expectiminimax algorithm** for decision-making.

### Methods:
1. **Heuristic Strategy:** A real-world inspired heuristic strategy for playing poker.
2. **Expectiminimax Algorithm:** A searching algorithm for decision-making in Texas Hold’em.

### Environment:
- **Python:** 3.8.13
- **Libraries:** 
  - `numpy==1.23.5`
  - `torch==2.3.0`
  - `scikit-learn==1.3.2`
  - `tensorflow==2.12.0`
  - `keras==2.12.0`
  - `pytorch_lightning==2.2.4`
  - `tqdm==4.66.4`

### Discussion:
I tested both approaches and analyzed their performance, discussing the pros and cons of each strategy in decision-making under uncertainty.
