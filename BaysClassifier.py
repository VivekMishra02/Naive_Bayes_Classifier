# Example of calculating class probabilities
from math import sqrt
from math import pi
from math import exp
import pandas as pd

# Split the dataset by class values, returns a dictionary

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    d = dataset
    for x in d:
        del(x[-1])
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*d)]
    #del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row, feature):
    probabilities = {'w1':1/2,'w2':1/2,'w3':0}
    for class_value, class_summaries in summaries.items():
        for i in range(len(class_summaries)-feature):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# Train calculating class probabilities
datasets = pd.read_csv("TrainData.csv")
dataset = datasets.values.tolist()

# Test calculating class probabilities
test_datasets = pd.read_csv("TestData.csv")
test_datasets = test_datasets.values.tolist()

summaries = summarize_by_class(dataset)

res1,res2,res3 = [],[],[]
count1,count2,count3 = 0,0,0

#For only X1

for x in range(len(test_datasets)):
    probabilities = calculate_class_probabilities(summaries, test_datasets[x],2)
    if probabilities["w1"] > probabilities["w2"]:
        res1.append("w1")
    else:
        res1.append("w2")

for x in range(len(test_datasets)):
    if res1[x] != test_datasets[x][3]:
        count1 +=1
print("ANS 3 B) The training error for only X1 is ",count1/len(test_datasets)*100,"%")

#For Only X1 and X2
for x in range(len(test_datasets)):
    probabilities = calculate_class_probabilities(summaries, test_datasets[x],1)
    if probabilities["w1"] > probabilities["w2"]:
        res2.append("w1")
    else:
        res2.append("w2")

for x in range(len(test_datasets)):
    if res2[x] != test_datasets[x][3]:
        count2 += 1
print("ANS 3 C) The training error for only X1 and X2 is ", count2 / len(test_datasets)*100,"%")

#For all the three features
for x in range(len(test_datasets)):
    probabilities = calculate_class_probabilities(summaries, test_datasets[x],0)
    if probabilities["w1"] > probabilities["w2"]:
        res3.append("w1")
    else:
        res3.append("w2")

for x in range(len(test_datasets)):
    if res2[x] != test_datasets[x][3]:
        count3 += 1
print("ANS 3 D) The training error for only X1, X2 and X3 is ", count3 / len(test_datasets) * 100, "%")

