###
# problem1_3.py
# perceptron for columbia edx artificial intelligence course
###
import sys
import csv

# load data
examples = []
with open(sys.argv[1], 'rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         examples.append(row)

# classifier function
def f(ex, w):
    s = 0
    for i in range(len(w)-1):
        s += int(ex[i]) * w[i]

    s += w[2]
    return (1 if s > 0 else -1)


def perceptron(examp):
    weights = [0,0,0]
    aWeights = []
    while (True):
        aWeights.append(weights[:])
        for ex in examp:
            if int(ex[2])*f(ex,weights) <= 0:
                weights[2] = weights[2] + int(ex[2])
                for i in range(len(weights)-1):
                    weights[i] = weights[i] + int(ex[2])*int(ex[i])

        if weights == aWeights[-1]:
            break
    return(weights, aWeights)



with open("output1.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    for w in perceptron(examples)[1]:
        wr.writerow(w)
