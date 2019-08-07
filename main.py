# Import required modules
import pandas as pd
import numpy as np


def data_setup():
    """
    Imports main_data and train_data.
    :return: A tuple that consists of main_data and train_data. Each of those tuples consists of a list of tuples,
    where each tuple defines the point in d-dimensional space. In the case of train_data, the last element of each
    tuple in the list denotes the class that the point belongs in, therefore each tuple's is d+1.
    """

    main_data = pd.read_csv('main_data.txt', sep=",", header=None)
    train_data = pd.read_csv('train_data.txt', sep=",", header=None)

    return [tuple(v) for v in main_data.values], [tuple(v) for v in train_data.values]


def dist_vect(item, train_data):
    """
    Computes the euclidean distance of one point (arg 1) from each point in train_data.
    :param item: Tuple that denotes the position of a point in the space
    :param train_data: List of tuples denoting the position of each point in train_data
    :return: Returns a list of tuples. The first element of k tuple in the list is the distance of given point (arg 1)
    from the k-th point of train_data, while the second element denotes the class of the k-th element of train_data.
    """

    distance_class = []  # initialize list to hold return list
    for element in train_data:
        dist = 0
        for j in range(len(item)):  # iterate all elements of tuple, except the last
            dist += (item[j] - element[j]) ** 2  # get squared distance of given point from current point in train_data

        current_class = element[len(item)]  # get class of current point
        distance_class.append((np.sqrt(dist), current_class))

    return distance_class


def dist_vectL1(item, train_data):
    """
    Computes the L1 norm between a given point (arg 1) and each point in train_data.
    :param item: Tuple that denotes the position of a point in the space
    :param train_data: List of tuples denoting the position of each point in train_data
    :return: Returns a list of tuples. The first element of k tuple in the list is the L1 distance of given point (arg 1)
    from the k-th point of train_data, while the second element denotes the class of the k-th element of train_data.
    """
    distance_class = []  # initialize list to hold return list
    for element in train_data:
        dist = 0
        for j in range(len(item)):  # iterate all elements of tuple, except the last
            dist += abs(item[j] - element[j])

        current_class = element[len(item)]  # get class of current point
        distance_class.append((dist, current_class))

    return distance_class


def dist_vectLinf(item, train_data):
    """
    Computes the L infinite norm between a given point (arg 1) and each point in train_data.
    :param item: Tuple that denotes the position of a point in the space
    :param train_data: List of tuples denoting the position of each point in train_data
    :return: Returns a list of tuples. The first element of k tuple in the list is the LInf distance of given point (arg 1)
    from the k-th point of train_data, while the second element denotes the class of the k-th element of train_data.
    """
    distance_class = []  # initialize list to hold return list
    for element in train_data:
        dist = []
        for j in range(len(item)):
            dist.append(abs(item[j] - element[j]))

        current_class = element[len(item)]  # get class of current point
        distance_class.append((max(dist), current_class))

    return distance_class


def decide_class(dv):
    """
    Decides the class of an item of main data based on majority. In case of a tie, the item is assigned acording to
    the class of its nearest neighbor.
    :param dv: List of tuples in which the first element of k tuple in the list is the distance of given point (arg 1)
    from the k-th point of train_data, while the second element denotes the class of the k-th element of train_data
    :return: Returns an integer indicating class of item.
    """
    dist, cl = zip(*dv)
    sort_i = np.argsort(dist)  # get the indexes that would sort the array
    number_of_neighbors = 3

    triplet = [dv[sort_i[i]][1] for i in range(number_of_neighbors)]  # get 3 nearest neighbours

    if len(set(triplet)) == 3:  # tie break - assign to the class of the nearest
        return int(cl[sort_i[0]])
    else:
        ballots = np.zeros([7, 1])  # create ballots for voting ( 0-based )
        for i in triplet:
            ballots[int(i)-1] += 1  # add one vote

        return ballots.argmax() + 1  # return the class with the more votes


def decide_class_wk(dv, k):
    """
    Decides the class of an item of main data based on weighted majority, using the inverse distance as weighting
    function.
    :param dv: List of tuples in which the first element of k tuple in the list is the distance of given point (arg 1)
    from the k-th point of train_data, while the second element denotes the class of the k-th element of train_data
    :param k: number of nearest neighbor to take into consideration.
    :return: Returns an integer indicating class of item.
    """
    dist, cl = zip(*dv)
    sort_i = np.argsort(dist)  # get the indexes that would sort the array

    ballots = np.zeros([7, 1])  # create ballots for voting
    for i in range(k):
        ballots[int(dv[sort_i[i]][1]) - 1] += 1 / dv[sort_i[i]][0]  # add weighted vote

    return ballots.argmax() + 1  # return the class with the more votes


main, train = data_setup()

# Initialize lists for tests
task2a = []
task2b1 = []
task2b2 = []
task2c3 = []
task2c8 = []
task2c25 = []

for ii in main:
    # Task 2A
    task2a.append(decide_class(dist_vect(ii, train)))

    # Task 2B
    task2b1.append(decide_class(dist_vectL1(ii, train)))
    task2b2.append(decide_class(dist_vectLinf(ii, train)))

    # Task 2C - Choose euclidean distance with weighted majority for 3, 8, 25 neighbors
    task2c3.append(decide_class_wk(dist_vect(ii, train), 3))
    task2c8.append(decide_class_wk(dist_vect(ii, train), 8))
    task2c25.append(decide_class_wk(dist_vect(ii, train), 25))

# Print results
print('Task 2A - Distance: Euclidean, Weigh Fun: Equal, NN: 3')
print(task2a)
print('Task 2B - Distance: L1-Norm, Weigh Fun: Equal, NN: 3')
print(task2b1)
print('Task 2B - Distance: LInf-Norm, Weigh Fun: Equal, NN: 3')
print(task2b2)
print('Task 2C - Distance: Euclidean, Weigh Fun: Inverse Distance, NN: 3')
print(task2c3)
print('Task 2C - Distance: Euclidean, Weigh Fun: Inverse Distance, NN: 8')
print(task2c8)
print('Task 2C - Distance: Euclidean, Weigh Fun: Inverse Distance, NN: 25')
print(task2c25)
