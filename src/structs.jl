#Result structure Definition : a struct that contains the resolution results that will be saved 
# Struct Definition

	mutable struct Results
		myDuration::Float64
		myGap::Float64
		isOptimal::Bool
		myTrain::DataFrames.DataFrame
		myTest::DataFrames.DataFrame
		myTestPrecision::Array{Float64, 2}
		myTestRecall::Array{Float64, 2}
		myTrainPrecision::Array{Float64, 2}
		myTrainRecall::Array{Float64, 2}
		function Results()
			return new()
		end
    end
	
	# Struct constructor
	function Results(duration::Float64, gap::Float64, optimal::Bool, train::DataFrames.DataFrame, test::DataFrames.DataFrame , testPrecision::Array{Float64, 2}, testRecall::Array{Float64, 2}, trainPrecision::Array{Float64, 2}, trainRecall::Array{Float64, 2})	
		this = Results()
		this.myDuration = duration
		this.myGap = gap
		this.isOptimal = optimal
		this.myTrain=train
		this.myTest=test
		this.myTestPrecision = testPrecision
		this.myTestRecall = testRecall
		this.myTrainPrecision = trainPrecision
		this.myTrainRecall = trainRecall
		return this
	end
	
	#number of ordered lists
	nb=2
	
	# Struct saving
	function Base.show(io::IO, myResult::Results)
		println(io, "Temps de résolution = ", myResults.myDuration)
		println(io, "Gap = ",myResults.myGap )
		println(io, "isOptimal = ", myResults.isOptimal )
		
		
		classTrainSize = Array{Int, 1}([0, 0])
		classTestSize = Array{Int, 1}([0, 0])
		nTrain = size(myResults.myTrain, 1)
		nTest = size(myResults.myTest, 1)
		for i in 1:nTrain
			if myResults.myTrain[i, 1] == 0
                classTrainSize[1] += 1
            else
                
                classTrainSize[2] += 1
            end 
		end
		
		for i in 1:nTest
			if myResults.myTest[i, 1] == 0
                classTestSize[1] += 1
            else
                
                classTestSize[2] += 1
            end 
		end
		
		for i in 1:nb
			if(i==1)
				println(io,"------------------ First order-----------------","\n")
			else
				println(io,"------------------ Second order-----------------","\n")
			end
			println(io,"-- Train results")
			println(io,"Class\tPrec.\tRecall")
			println(io,"0\t", round(myResults.myTrainPrecision[1, i], digits=2), "\t", round(myResults.myTrainRecall[1, i], digits=2), "\t", classTrainSize[1])
			println(io,"1\t", round(myResults.myTrainPrecision[2, i], digits=2), "\t", round(myResults.myTrainRecall[2, i], digits=2), "\t", classTrainSize[2],"\n")
			println(io,"avg\t", round((myResults.myTrainPrecision[1, i] + myResults.myTrainPrecision[2, i])/2, digits=2), "\t", round((myResults.myTrainRecall[1, i] + myResults.myTrainRecall[2, i])/2, digits=2))
			println(io,"w. avg\t", round(myResults.myTrainPrecision[1, i] * classTrainSize[1] / nTrain + myResults.myTrainPrecision[2, i] * classTrainSize[2] / nTrain , digits = 2), "\t", round(myResults.myTrainRecall[1, i] * classTrainSize[1] / nTrain + myResults.myTrainRecall[2, i] * classTrainSize[2] /nTrain , digits = 2), "\n")
    
			println(io,"-- Test results")
			println(io,"Class\tPrec.\tRecall")
			println(io,"0\t", round(myResults.myTestPrecision[1, i], digits=2), "\t", round(myResults.myTestRecall[1, i], digits=2), "\t", classTestSize[1])
			println(io,"1\t", round(myResults.myTestPrecision[2, i], digits=2), "\t", round(myResults.myTestRecall[2, i], digits=2), "\t", classTestSize[2],"\n")
			println(io,"avg\t", round((myResults.myTestPrecision[1, i] + myResults.myTestPrecision[2, i])/2, digits=2), "\t", round((myResults.myTestRecall[1, i] + myResults.myTestRecall[2, i])/2, digits=2))
			println(io,"w. avg\t", round(myResults.myTestPrecision[1, i] * classTestSize[1] / nTest + myResults.myTestPrecision[2, i] * classTestSize[2] / nTest , digits = 2), "\t", round(myResults.myTestRecall[1, i] * classTestSize[1] / nTest + myResults.myTestRecall[2, i] * classTestSize[2] /nTest , digits = 2), "\n")
		
		end
	end
	