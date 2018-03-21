# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def confusionMatrix(testSet, predictions):
    #TP = true positive, FP = false positive
    #TN = true negative, FN = false negative
    TP, FP, TN, FN = 0,0,0,0
    for i in range(len(testSet)):
        if testSet[i][-1] == 1:
            if testSet[i][-1] == predictions[i]:
                TP +=1
            else:
                FN +=1
        else:
            if testSet[i][-1] == predictions[i]:
                TN +=1
            else:
                FP +=1
    return[TP, FN, FP, TN]

def precision(confussionMatrix):
    precission = confussionMatrix[0]/(confussionMatrix[0]+confussionMatrix[2])
    return precission

def recall(confussionMatrix):
    recall = confussionMatrix[0] / (confussionMatrix[0] + confussionMatrix[1])
    return recall


def main():
    filename = 'pima-indians-diabetes.data.csv'# For Each Attribute:(all numeric-valued)
    #                                             1. Number of times pregnant
    #                                             2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    #                                             3. Diastolic blood pressure (mm Hg)
    #                                             4. Triceps skin fold thickness (mm)
    #                                             5. 2-Hour serum insulin (mu U/ml)
    #                                             6. Body mass index (weight in kg/(height in m)^2)
    #                                             7. Diabetes pedigree function
    #                                             8. Age (years)
    #                                             9. Class variable (0 or 1)
    #
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    confustionMatrixResult = confusionMatrix(testSet,predictions)
    precisionResult = precision(confustionMatrixResult)
    recallResult = recall(confustionMatrixResult)
    print('Accuracy: {0}%'.format(accuracy))
    print("Confusion Matrix:" + str(confustionMatrixResult))
    print("Precision: " + str(precisionResult))
    print("Recall: " + str(recallResult))

    model = GaussianNB()
    dataset = datasets.load_diabetes()
    model.fit(dataset.data, dataset.target)
    expected = dataset.target
    predicted = model.predict(dataset.data)
    print("Library one:")
    print(metrics.classification_report(expected,predicted))
    print(metrics.confusion_matrix(expected,predicted))


main()