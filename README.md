# PythonProgrammingTask1
First assignment of Python programming module for the Computational Mathematical Finance UoE

## Task 2 - K-Nearest-neighbours
### Task 2a - First implementation
For this task you will need to implement a [K-nearest-neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithm for classification from scratch (**important - do not use a version of this algorithm from another module e.g SKLearn - you need to write the functions as directed yourself**). The K-nearest-neighbours algorithm is a classic machine learning algorithm used for [classification problems](https://en.wikipedia.org/wiki/Statistical_classification). Classifying a data item from a given dataset means deciding which of a number of classes the item belongs to. This is done using a *training set*, containing a number of data items with known class.

We will consider first a simple implementation of the 3-nearest-neighbour version of the algorithm. The algorithm proceeds as follows.

For each item in the dataset, we find the 3 nearest items (and their respective classes) in the training set. We then attribute a class to our data item by majority voting amongst the classes of the 3 nearest neighbours. In the case where three classes are tied in the vote, we resolve the tie by choosing the class of whichever neighbour is the nearest in the training set. (In practice this might not be a very good thing to do - but we are just constructing a simple implementation for now). Finally, we need to output the class we have decided on for each item in our original dataset.

The data is contained in two files `main_data.txt` and `train_data.txt`.
* Each item in `main_data.txt` must be classified into one of 7 classes, using the training set `train_data.txt`.
* Each line of `train_data.txt` defines 1 training item, as a vector of floating-point values, followed by its class label (a single integer in the range 1-7).
* Each line of `main_data.txt` defines 1 data item, as only a vector of floating-point values, without a class label.

To do this write 3 functions:

- **`data_setup()`** which reads in both the dataset contained in `main_data.txt`, and the training dataset contained in `train_data.txt` into two lists - returning the two lists (`main_data`, and `train_data`) from the function. The lists containing each item of the dataset should use appropriate types for each element of the list. 
- **`dist_vect(item,train_data)`** which takes as arguments one element of the list `main_data`, and the list `train_data`, both from the output of the data_setup function. The function should calculate the Euclidean distance from item in the dataset to each item in the training dataset in order. The function should return a list of (distance, class) tuples.
- **`decide_class(dv)`** which takes the output list of tuples of the dist_vect function as an argument, and uses it to return the class of the item in the dataset.

- Lastly use your functions to obtain a list containing the calculated class for each of the element in the `main_data` list you have constructed.

### Task 2b - Changing the distance measure
Next write two different replacements for the **dist_vect(item,train_data)** function. The function you have written above calculates the distance between data items using Euclidean distance as:
$$d(x,y) = \left(\sum_{i=1}^n (x_i-y_i)^2\right)^{1/2}$$

For this task write two more versions of the distance function using different distance metrics.

- Firstly **dist_vectL1(item,train_data)** where distance is determined by using:

$$d(x,y) = \sum_{i=1}^n \left|x_i-y_i\right|$$

- Next, **dist_vectLinf(item,train_data)** where distance is determined by using:

$$d(x,y) = \max\limits_{i=1..n}\left|x_i-y_i\right|$$

These are the $L_1$ and $L_\infty$ norms respectively. 

- Lastly use the new functions to obtain a list containing the calculated class for each of the element in the main_data list you have constructed.

### Task 2c - Changing the decision methodology
The simple majority-voting implemented in your `decide_class()` function is unweighted --- once the 3 nearest neighbours are found, they each have 1 vote, regardless of which is closest (unless there is a tie). An alternative is *weighted voting*, where each of the $k$ nearest neighbours is attributed a coefficient, such that the closest neighbours carry more weight in the result.

Write a function `decide_class_wk(dv,k)` which also takes the number $k$ of nearest neighbours as an input argument. Your function should attribute a weight $w_j$ to each of the $k$ nearest neighbours, inversely proportional to their distance from the data item:

$$
w_j = \frac{1}{d_j(x,y)}\, , \quad j \in \{1,\ldots,k\}
$$

The score for each class is determined by the sum of the weights of each item in that class amongst the $k$ nearest neighbours. The class with the highest score is chosen for the data item.

(Note that the simple majority-voting corresponds to setting all the weights to 1.)

Test your function on the dataset, for $k=3, 8,$ and $25$.
