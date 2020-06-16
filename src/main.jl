using CSV
using JuMP
using CPLEX
using DataFrames
using Random

include("functions.jl")
include("structs.jl")

dataSet = "heart"
dataFolder = "../data/"
resultsFolder = "../res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
# Details:
# - read the file ./data/kidney.csv
# - save the features in ./data/kidney_test.csv and ./data/kidney_train.csv
train, test = createFeatures(dataFolder, dataSet)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
# Details:
# - read the file ./data/kidney_train.csv
# - save the rules in ./res/kidney_rules.csv
rules = createRules(dataSet, resultsFolder, train)

# Order the rules (limit the resolution to 300 seconds)
# Details:
# - read the file ./data/kidney_rules.csv
# - save the rules in ./res/kidney_ordered_rules.csv
timeLimitInSeconds = 300
orderedRules1, orderedRules2, optimal, gap, duration  = sortRules(dataSet, resultsFolder, train, rules, timeLimitInSeconds)

nb=2
trainPrecision = zeros(2, nb)
trainRecall = zeros(2, nb)
testPrecision = zeros(2, nb)
testRecall = zeros(2, nb)

println("------------------ First order-----------------", "\n")
println("-- Train results")
trainPrecision[:, 1], trainRecall[:, 1] = showStatistics(orderedRules1, train)

println("-- Test results")
testPrecision[:, 1], testRecall[:, 1] = showStatistics(orderedRules1, test)
println(" ")
println("------------------ Second order-----------------","\n")

println("-- Train results")
trainPrecision[:, 2], trainRecall[:, 2] = showStatistics(orderedRules2, train)

println("-- Test results")
testPrecision[:, 2], testRecall[:, 2] = showStatistics(orderedRules2, test)



myResults=Results(duration, gap, optimal, train, test, testPrecision, testRecall, trainPrecision, trainRecall)
saveResults(dataSet, resultsFolder, myResults)