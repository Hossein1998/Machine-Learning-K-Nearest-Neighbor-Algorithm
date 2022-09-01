import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from random import randrange
from collections import Counter

df = pd.read_csv('uspsdata.csv' ,sep="\t")
lbl = pd.read_csv('uspscl.csv' ,sep="\t")

arr = np.array(df)
arrlbl=np.array(lbl)


a_3d_array = np.zeros((200,16,16))

for i in range(0,199):
    
    a_3d_array[i]=(np.array(df.iloc[i])).reshape(16, 16)
    


arr1=np.array(df.iloc[1])
newarr = arr1.reshape(16, 16)   
plt.imshow(newarr, interpolation='nearest')


arr2=np.array(df.iloc[2])
newarr = arr2.reshape(16, 16)   
plt.imshow(newarr, interpolation='nearest')


arr3=np.array(df.iloc[3])
newarr = arr3.reshape(16, 16)   
plt.imshow(newarr, interpolation='nearest')
 
        
arr4=np.array(df.iloc[4])
newarr = arr4.reshape(16, 16)   
plt.imshow(newarr, interpolation='nearest') 
 
 
            


df = df.assign(lbls=lbl.values)


train, validate, test = np.split(df.sample(frac=1), [int(.6*len(a_3d_array)), int(.8*len(a_3d_array))])
TrainArray=np.array(train)
ValidateArray=np.array(validate)
TestArray=np.array(test)
TrainDataset = list()
TestDataset = list()
for row in train:
			if not row:
				continue
			TrainDataset.append(row)
for row in test:
			if not row:
				continue
			TestDataset.append(row)

# print(TestDataset)   
# print(test)  
# print(TestArray)


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def get_error_rate(training_dataset, test_dataset,k):
     
    errors_count = 0
    for test_data in test_dataset:
        # classify each data in test dataset
        predicted_class = predict_classification(training_dataset, test_data,k)

        if predicted_class != test_data[256]:
            errors_count += 1

    # return errors_count / test_dataset_count as error rate
    return errors_count / len(test_dataset)


k=2
error=get_error_rate(TrainArray,TestArray,k)
print('error: %s' % error)
 


