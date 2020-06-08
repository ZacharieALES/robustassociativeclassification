# Tolerance
epsilon = 0.0001

"""
Function used to transform a column with numerical values into one or several binary columns.

Arguments:
 - data: table which contains the column that will be binarized (1 row = 1 individual, 1 column = 1 feature);
 - header: header of the column of data that will be binarized
 - intervals: array of values which delimits the binarization (ex : [2, 4, 6, 8] will lead to 3 columns respectively equal to 1 if the value of column "header" is in [2, 3], [4, 5] and [6, 7])

 Example:
  createColumns(:Age, [1, 17, 50, Inf], data, features) will create 3 binary columns in features named "Age1-16", "Age17-49", "Age50-Inf"
"""
function createColumns(header::Symbol, intervals, data::DataFrames.DataFrame, features::DataFrames.DataFrame)
    for i in 1:size(intervals, 1) - 1
        lb = intervals[i]
        ub = intervals[i+1]
        features[!, Symbol(header, lb, "-", (ub-1))] = ifelse.((data[!, header] .>= lb) .& (data[!, header] .< ub), 1, 0) 
    end
end

"""
Create the train and test csv files of a data set

Arguments:
 - dataFolder: folder which contains the data set csv file (ex: "./data")
 - dataSet: name of the data set (ex: "titanic")

Important remark: the first column of the output files must correspond to the class of each individual (and it must be 0 or 1)
"""
function createFeatures(dataFolder::String, dataSet::String)

    # Get the input file path
    rawDataPath = dataFolder * dataSet * ".csv"

    # Test its existence
    if !isfile(rawDataPath)
        println("Error in createFeatures: Input file not found: ", rawDataPath)
        return
    end

    # Put the data in a DataFrame variable
    rawData = CSV.read(rawDataPath,  header=true)

    # Output files path
    trainDataPath = dataFolder * dataSet * "_train.csv"
    testDataPath = dataFolder * dataSet * "_test.csv"

    # If the train or the test file do not exist
    if !isfile(trainDataPath) || !isfile(testDataPath)

        println("=== Creating the features")

        # Create the table that will contain the features
        features::DataFrame = DataFrames.DataFrame()
        
        # Create the features of the titanic data set
        if dataSet == "titanic"

            # Add the column related to the class (always do it first!)
            # Remark: here the rawData already contain 0/1 values so no change needed
            features.Survived = rawData.Survived

            #### First example of the binarization of a column with numerical values
            # Add columns related to the ages
            # -> 3 columns ([0, 16], [17, 49], [50, +infinity[)
            # Ex : if the age is 20, the value of these 3 columns will be 0, 1 and 0, respectively.
            createColumns(:Age, [0, 17, 50, Inf], rawData, features)
            
            # Add columns related to the fares
            # -> 3 columns ([0, 9], [10, 19], [20, +infinity[)
            createColumns(:Fare,  [0, 10, 20, Inf], rawData, features)
            
            #### First example of the binarization of a column with categorical values
            # Add 1 column for the sex (female or not)
            # Detailed description of the command:
            # - create in DataFrame "features" a column named "Sex"
            # - for each row of index i of "rawData", if column "Sex" is equal to "female", set the value of column "Sex" in row i of features to 1; otherwise set it to 0
            #features.Sex = ifelse.(rawData.Sex .== "female", 1, 0)
            
            # Add columns related to the passenger class
            # -> 3 columns (class 1, class 2 and class 3)
            
            # For each existing value in the column "Pclass"
            #for a in sort(unique(rawData.Pclass))

                # Create 1 feature column named "Class1", "Class2" or "Class3"
                #features[!, Symbol("Class", a)] = ifelse.(rawData.Pclass .<= a, 1, 0)
            #end

            # Add a column related  to the number of relatives
            # -> 1 column (0: no relatives, 1: at least one relative)
            #features.Relative = ifelse.(rawData[!, Symbol("Siblings/Spouses Aboard")] + rawData[!, Symbol("Parents/Children Aboard")] .> 0, 1, 0)


        end

        if dataSet == "kidney"

            features.class = ifelse.(rawData.class .== "notckd", 0, 1)
            createColumns(:age, [0, 35, 55, Inf], rawData, features)
            createColumns(:bp, [0, 70, 90, Inf], rawData, features)
            features.al = ifelse.(rawData.al .== "0", 0, 1)

        end 
		
		if dataSet == "heart"
			features.Class = ifelse.(rawData.Class .== 1, 0, 1)
			createColumns(:Age, [0,35,50, Inf], rawData, features)
			features.Sex = rawData.Sex
			
			# Add columns related to the Chest Pain type
            # -> 4 columns (type 1, type 2, type 3 and type 4)
            
            # For each existing value in the column "ChestPainType
            for a in sort(unique(rawData.ChestPainType))

                # Create 1 feature column named "ChestPainType1", "ChestPainType2", "ChestPainType3" or "ChestPainType4"
                features[!, Symbol("ChestPainType", a)] = ifelse.(rawData.ChestPainType .== a, 1, 0)
            end
			#createColumns(:RestingBloodPressure, [0, 120, 140, 160, 180, Inf], rawData, features)
			createColumns(:RestingBloodPressure, [0,  120,  160, Inf], rawData, features)
			createColumns(:SerumCholestoral, [0, 200, 240, Inf], rawData, features)
			features.FastingBloodSugar = ifelse.(rawData.FastingBloodSugar .> 120, 1, 0)
			
			# Add columns related to the Resting Electrocardiographic Results
            # -> 3 columns (RestingECG1, RestingECG2 and RestingECG3)
            
			# For each existing value in the column "RestingECGResults"
            for a in sort(unique(rawData.RestingECGResults))

                # Create 1 feature column named "Class1", "Class2" or "Class3"
                features[!, Symbol("RestingECG", a)] = ifelse.(rawData.RestingECGResults .== a, 1, 0)
            end
			
			#createColumns(:MaximumHeartRate, [0, 150, 160, 170, 180, Inf], rawData, features)
			createColumns(:MaximumHeartRate, [0, 160, 180, Inf], rawData, features)
			#features.ExerciseInducedAngina = rawData.ExerciseInducedAngina
			createColumns(:Oldpeak, [0, 0.8, 1.8, 6.8], rawData, features)
			
			# Add columns related to the slope of the peak exercise ST segment
            # -> 3 columns (STSlope1, STSlope2 and STSlope3)
            
			# For each existing value in the column "STSlope"
            #for a in sort(unique(rawData.STSlope))

                # Create 1 feature column named "Class1", "Class2" or "Class3"
                #features[!, Symbol("STSlope", a)] = ifelse.(rawData.STSlope .<= a, 1, 0)
            #end
			
			# Add columns related to the number of major vessels (0-3) colored by flourosopy
            # -> 3 columns (MajorVessels1, MajorVessels2 and MajorVessels3)
            
			# For each existing value in the column "MajorVesselsNumber"
            #for a in sort(unique(rawData.MajorVesselsNumber))

                # Create 1 feature column named "Class1", "Class2" or "Class3"
                #features[!, Symbol("MajorVessels", a)] = ifelse.(rawData.MajorVesselsNumber .<= a, 1, 0)
            #end
			
			# Add columns related to the thal
            # -> 3 columns (normal, fixed defect, reversable defect)
   			#features[!, Symbol("NormalThal")] = ifelse.(rawData.Thal .== 3, 1, 0)
			#features[!, Symbol("FixedDefectThal")] = ifelse.(rawData.Thal .== 6, 1, 0)
            #features[!, Symbol("ReversableDefectThal")] = ifelse.(rawData.Thal .== 7, 1, 0)
		end
		
        
        if dataSet == "other"
            #TODO
        end 

        # Shuffle the individuals
        features = features[shuffle(1:size(features, 1)),:]

		
	
        # Split them between train and test
		# Split them between train and test
		featuresC0=features[features[:,1] .== 0, :]
		featuresC1=features[features[:,1] .== 1, :]
		LimitC0 = trunc(Int, size(featuresC0, 1) * 1/2)
		LimitC1 = trunc(Int, size(featuresC1, 1) * 1/2)
		if(mod(LimitC0,2) !=0)
			LimitC0+=mod(LimitC0,2)
		end
		if(mod(LimitC1,2) !=0)
			LimitC1+=mod(LimitC1,2)
		end
		train = vcat(featuresC0[1:LimitC0, :] , featuresC1[1:LimitC1, :])
		test = vcat(featuresC0[(LimitC0+1):end, :] , featuresC1[(LimitC1+1):end, :])
        
        CSV.write(trainDataPath, train)
        CSV.write(testDataPath, test)

    # If the train and test file already exist
    else
        println("=== Warning: Existing features found, features creation skipped")
        println("=== Loading existing features")
        train = CSV.read(trainDataPath)
        test = CSV.read(testDataPath)
    end
    
    println("=== ... ", size(train, 1), " individuals in the train set")
    println("=== ... ", size(test, 1), " individuals in the test set")
    println("=== ... ", size(train, 2), " features")
    
    return train, test
end 


"""
Create the association rules related to a training set

Arguments
 - dataSet: name of the data ste
 - resultsFolder: name of the folser in which the rules will be written
 - train: DataFrame with the training set (each line is an individual, the first column is the class, the other are the features)

Output
 - table of rules (each line is a rule, the first column corresponds to the rules class)
"""
function createRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame)

    # Output file
    rulesPath = resultsFolder * dataSet * "_rules.csv"
    rules = []

    if !isfile(rulesPath)

        println("=== Generating the rules")
        
        # Transactions
        t::DataFrame = train[:, 2:end]

        # Class of the transactions
        # Help: to get the class of transaction nb i:
        # - do not use: transactionClass[i]
        # - use: transactionClass[i, 1]
        transactionClass::DataFrame = train[:, 1:1]

        # Number of features
        d::Int64 = size(t, 2)

        # Number of transactions
        n::Int64 = size(t, 1)

        mincovy::Float64 = 0.05
        iterlim::Int64 = 5
        RgenX::Float64 = 0.1 / n
        RgenB::Float64 = 0.1 / (n * d)
        
        ##################
        # Find the rules for each class
        ##################
        for y = 0:1

            println("-- Classe $y")

            # Help: Let rule be a rule that you want to add to rules
            # - if it is the first rule, use: rules = rule
            # - if it is not the first rule, use: rules = append!(rules, rule)
            
            sxb::Float64 = n
            iter::Int64 = 1
            sb::Float64 = 0

            ##################
            # Define the model
            ##################

            # Create the model
            m = Model(with_optimizer(CPLEX.Optimizer))

            # Remove CPLEX output
            set_parameter(m, "CPX_PARAM_SCRIND", 0)

            @variable(m, 0 <= x[i in 1:n] <= 1)
            @variable(m, b[j in 1:d], Bin)

            # Cstr 1
            @constraint(m, [i = 1:n, j = 1:d], x[i] <= 1 + (t[i,j]-1) * b[j])

            # Cstr 2
            @constraint(m, [i = 1:n], x[i] >= 1 + sum((t[i,j]-1) * b[j] for j = 1:d))

            # Cstr 5
            @constraint(m, sum(x[i] for i in 1:n) <= sxb)
            
            # Set the objective
            @objective(m, Max, sum(x[i] for i = 1:n if transactionClass[i, 1] == y)
                       - RgenX * sum(x[i] for i = 1:n)
                       - RgenB * sum(b[j] for j = 1:d))

            isOver = false;
            while(!isOver)

                if iter == 1
                    println("- Solve iter $iter for sxb = $sxb")
                    optimize!(m)

                    sb = sum(JuMP.value(x[i]) for i = 1:n if transactionClass[i, 1] == y)
                    iter += 1
                end
                
                # Add the new rule
                rule = convert(DataFrame, hcat(append!([y], trunc.(Int, JuMP.value.(b)))...))

                @show typeof(rule)
                @show typeof(rules)
                if size(rules, 1) > 0
                    append!(rules, rule)
                else
                    rules = rule
                end 

                # Add the new constraint (to avoid getting the same rule again)
                @constraint(m, sum(b[i] for i in 1:d if JuMP.value(b[i]) < epsilon)
                            +  sum(1 - b[i] for i in 1:d if JuMP.value(b[i]) > 1 - epsilon)
                            >= 1)

                if iter <= iterlim

                    println("- Solve iter $iter for sxb = $sxb")
                    optimize!(m)
                    
                    if sum(JuMP.value(x[i]) for i = 1:n if transactionClass[i, 1] == y) < sb
                        sxb = min(sum(JuMP.value(x[i]) for i = 1:n), sxb - 1)
                        @constraint(m, sum(x[i] for i in 1:n) <= sxb) 
                        iter = 1
                    else
                        iter += 1
                    end
                else
                    sxb -= 1
                    @constraint(m, sum(x[i] for i in 1:n) <= sxb) 
                    iter = 1
                end
                
                if sxb < n * mincovy
                    isOver = true
                end
            end
        end
        
        CSV.write(rulesPath, rules)

    else
        println("=== Warning: Existing rules found, rules creation skipped")
        println("=== Loading the existing rules")
        rules = CSV.read(rulesPath)
    end
    
    println("=== ... ", size(rules, 1), " rules obtained") 

    return rules
end

"""
Sort the rules

Arguments
  - dataSet: name of the dataset folder
  - resultsFolder: name of the folder in which the results are written
  - train: train data set (1 row = 1 individual)
  - rules: rules which must be sorted (1 row = 1 rule)
  - tilim: maximal running time of CPLEX in seconds
"""
function sortRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame, rules::DataFrames.DataFrame, tilim::Int64)

    orderedRulesPath1 = resultsFolder * dataSet * "_ordered_rules1.csv"
    orderedRulesPath2 = resultsFolder * dataSet * "_ordered_rules2.csv"
    if !isfile(orderedRulesPath1) || !isfile(orderedRulesPath2)

        println("=== Sorting the rules")
		
		#number of train data sets
		nb=2
		
		#proximity approach
		#approach=1
		approach=2
		
		#Split the train into two sets
		trainC0=train[train[:,1] .== 0, :]
		trainC1=train[train[:,1] .== 1, :]
		LimitC0 = trunc(Int, size(trainC0, 1) * 1/2)
		LimitC1 = trunc(Int, size(trainC1, 1) * 1/2)
		train1 = vcat(trainC0[1:LimitC0, :] , trainC1[1:LimitC1, :])
		train2 = vcat(trainC0[(LimitC0+1):end, :] , trainC1[(LimitC1+1):end, :])
		println("=== ... ", size(train1, 1), " train1")
        println("=== ... ", size(train2, 1), " train2")
    
        # Transactions
        t1 = train1[:, 2:end]
		t2 = train2[:, 2:end]

        # Class of the transactions
        transactionClass1 = train1[:, 1:1]
		transactionClass2 = train2[:, 1:1]

        # Number of features
        d = size(t1, 2)

        # Number of transactions in each train data set
        n = size(t1, 1)

        # Add the two null rules in first position
        nullrules = similar(rules, 0)
        push!(nullrules, append!([0], zeros(d)))
        push!(nullrules, append!([1], zeros(d)))
        rules = vcat(nullrules, rules)

        # Remove duplicated rules
        rules = unique(rules)

        # Number of rules
        L = size(rules)[1]

        Rrank = 1/L

        ################
        # Compute the v^q_il and p^q_il constants
        # for each train data set q , p^q_il = :
        #  0 if rule l does not apply to transaction i
        #  1 if rule l applies to transaction i and   correctly classifies it
        # -1 if rule l applies to transaction i and incorrectly classifies it
        ################
		p = zeros(n, L, nb)
		p[:, :, 1]=createPi(t1, rules, transactionClass1)
		p[:, :, 2]=createPi(t2, rules, transactionClass2)
		
        v = abs.(p)
		
        ################
        # Create and solve the model
        ###############
        #CPX_PARAM_SCRIND=0
        m = Model(with_optimizer(CPLEX.Optimizer))
        set_parameter(m, "CPX_PARAM_TILIM", tilim)
        
        # u^q_il: rule l is the highest which applies to transaction i in train data set q
        @variable(m, u[1:n, 1:L, 1:nb], Bin)
		
        # r^q_l: rank of rule l in train data set q
        @variable(m, 1 <= r[1:L, 1:nb] <= L, Int)

        # rstar^q: rank of the highest null rule in train data set q
        @variable(m, 1 <= rstar[1:nb] <= L)
        @variable(m, 1 <= rB[1:nb] <= L)

        # g^q_i: rank of the highest rule which applies to transaction i in train data set q
        @variable(m, 1 <= g[1:n, 1:nb] <= L, Int)
		
        # s^q_lk: rule l is assigned to rank k in train data set q
        @variable(m, s[1:L, 1:L,1:nb], Bin)
		
		
        # Rank of null rules in each train data set q
        rA = r[1, :]
        rB = r[2, :]


        # rstar^q == rB^q?
        @variable(m, alpha[1:nb], Bin)
		
        # rstar^q == rA^q?
        @variable(m, 0 <= beta[1:nb] <= 1)

        # Maximize the classification accuracy for each train data set q
        @objective(m, Max, sum(p[i, l, 1] * u[i, l, 1] for i in 1:n for l in 1:L)
                   + Rrank * rstar[1] + sum(p[i, l, 2] * u[i, l, 2] for i in 1:n for l in 1:L)
                   + Rrank * rstar[2])

        # For each train data set q, Only one rule is the highest which applies to transaction i
        @constraint(m, [i in 1:n, q in 1:nb], sum(u[i, l, q] for l in 1:L) == 1)
		
		
        # g constraints
        @constraint(m, [i in 1:n, l in 1:L, q in 1:nb], g[i, q] >= v[i, l, q] * r[l, q])
        @constraint(m, [i in 1:n, l in 1:L, q in 1:nb], g[i, q] <= v[i, l, q] * r[l, q] + L * (1 - u[i, l, q]))
		
        # Relaxation improvement
        @constraint(m, [i in 1:n, l in 1:L, q in 1:nb], u[i, l, q] >= 1 - g[i, q] + v[i, l, q] * r[l, q])
        @constraint(m, [i in 1:n, l in 1:L, q in 1:nb], u[i, l, q] <= v[i, l, q]) 
		
        # r constraints
        @constraint(m, [k in 1:L, q in 1:nb], sum(s[l, k, q] for l in 1:L) == 1)
        @constraint(m, [l in 1:L, q in 1:nb], sum(s[l, k, q] for k in 1:L) == 1)
        @constraint(m, [l in 1:L, q in 1:nb], r[l, q] == sum(k * s[l, k, q] for k in 1:L))
	

        # rstar constraints
        @constraint(m, [q in 1:nb], rstar[q] >= rA[q])
        @constraint(m, [q in 1:nb], rstar[q] >= rB[q])
        @constraint(m, [q in 1:nb], rstar[q] - rA[q] <= (L-1) * alpha[q])
        @constraint(m, [q in 1:nb], rA[q] - rstar[q] <= (L-1) * alpha[q])
        @constraint(m, [q in 1:nb], rstar[q] - rB[q] <= (L-1) * beta[q])
        @constraint(m, [q in 1:nb], rB[q] - rstar[q] <= (L-1) * beta[q])
        @constraint(m, [q in 1:nb], alpha[q] + beta[q] == 1)
		
		

        # u^q_il == 0 if rstar^q > r^q_l (also improve relaxation)
        @constraint(m, [i in 1:n, l in 1:L, q in 1:nb], u[i, l, q] <= 1 - (rstar[q] - r[l, q])/ (L - 1))
		
		
		
		#Proximity of two lists r1 and r2
		    
		if(approach == 1)
			#First approach
			@variable(m, x1[1:L, 1:L], Bin)
			@variable(m, x2[1:L, 1:L], Bin)
			@variable(m, x[1:L, 1:L], Bin)
			@constraint(m, [i in 1:L, j in 1:L], x1[i, j] * L >= r[j, 1]-r[i, 1])
			@constraint(m, [i in 1:L, j in 1:L], x2[i, j] * L >= r[j, 2]-r[i, 2])
			@constraint(m, [i in 1:L, j in 1:L], x[i, j] >= x1[i, j] + x2[i, j]-1)
			@constraint(m, [i in 1:L, j in 1:L], x[i, j] <= x1[i, j] )
			@constraint(m, [i in 1:L, j in 1:L], x[i, j] <= x2[i, j] )
			@constraint(m, sum(x[i, j] for i in 1:L for j in 1:L) <= 2 * L)
		else
			#Second approach
			@variable(m, 0 <= y[1:L] <= L-1, Int)
			@constraint(m, [i in 1:L], y[i]  >= r[i, 1]-r[i, 2])
			@constraint(m, [i in 1:L], y[i]  >= r[i, 2]-r[i, 1])
			@constraint(m, sum(y[i] for i in 1:L ) <= 2 * L)
		end
			
        status = optimize!(m)

        ###############
        # Write the rstar highest ranked rules and their corresponding class
        ###############

        # Number of rules kept in the classifier
        # (all the rules ranked lower than rstar are removed)
        relevantNbOfRules1=L-trunc(Int, JuMP.value(rstar[1]))+1
		relevantNbOfRules2=L-trunc(Int, JuMP.value(rstar[2]))+1

        # Sort the rules and their class by decreasing rank
        rulesOrder1 = JuMP.value.(r[:, 1])
		rulesOrder2 = JuMP.value.(r[:, 2])
        orderedRules1 = rules[sortperm(L.-rulesOrder1), :]
		orderedRules2 = rules[sortperm(L.-rulesOrder2), :]
		
        orderedRules1 = orderedRules1[1:relevantNbOfRules1, :]
		orderedRules2 = orderedRules2[1:relevantNbOfRules2, :]
		
        CSV.write(orderedRulesPath1, orderedRules1)
		CSV.write(orderedRulesPath2, orderedRules2)

    else
        println("=== Warning: Sorted rules found, sorting of the rules skipped")
        println("=== Loading the sorting rules")
        orderedRules1 = CSV.read(orderedRulesPath1)
		orderedRules2 = CSV.read(orderedRulesPath2)
    end 

    #return orderedRules

end


function createPi(t::DataFrames.DataFrame, rules::DataFrames.DataFrame, transactionClass::DataFrames.DataFrame)

	# Number of features
    d::Int64 = size(t, 2)

    # Number of transactions
    n::Int64 = size(t, 1)
	
	# Number of rules
    L = size(rules)[1]
		
	p = zeros(n, L)
	
	# For each transaction and each rule
	for i in 1:n
		for l in 1:L
		# If rule l applies to transaction i
		# i.e., if the vector t_i - r_l does not contain any negative value
			if !any(x->(x<-epsilon), [sum(t[i, k]-rules[l, k+1]) for k in 1:d])

				# If rule l correctly classifies transaction i
				if transactionClass[i, 1] == rules[l, 1]
					p[i, l] = 1
				else
					p[i, l] = -1 
				end
			end
		end
	end
	return p
end




"""
Compute for a given data set the precision and the recall of 
- each class
- the whole data set (with and without weight for each class)

Arguments
  - orderedRules: list of rules of the classifier (1st row = 1st rule to test)
  - dataset: the data set (1 row = 1 individual)
"""
function showStatistics(orderedRules::DataFrames.DataFrame, dataSet::DataFrames.DataFrame)

    
    # Number of transactions
    n = size(dataSet, 1)

    # Statistics with respect to class 0:
    # - true positive;
    # - true negative;
    # - false positive;
    # - false negative
    tp::Int = 0
    fp::Int = 0
    fn::Int = 0
    tn::Int = 0

    # Number of individuals in each class
    classSize = Array{Int, 1}([0, 0])
    
    # For all transaction i in the data set
    for i in 1:n

        # Get the first rule satisfied by transaction i
        ruleId = findfirst(all, collect(eachrow(Array{Float64, 2}(orderedRules[:, 2:end])  .<= Array{Float64, 2}(DataFrame(dataSet[i, 2:end])))))

        # If transaction i is classified correctly (i.e., if it is a true)
        if orderedRules[ruleId, 1] == dataSet[i, 1]

            # If transaction i is of class 0
            if dataSet[i, 1] == 0
                tp += 1
                classSize[1] += 1
            else
                tn += 1
                classSize[2] += 1
            end 

            # If it is a negative
        else

            # If transaction i is of class 0
            if dataSet[i, 1] == 0
                fn += 1
                classSize[1] += 1
            else
                fp += 1
                classSize[2] += 1
            end 
        end
    end
    
    precision = Array{Float64, 1}([tp / (tp+fp), tn / (tn+fn)])
    recall = Array{Float64, 1}([tp / (tp + fn), tn / (tn + fp)])

    println("Class\tPrec.\tRecall\tSize")
    println("0\t", round(precision[1], digits=2), "\t", round(recall[1], digits=2), "\t", classSize[1])
    println("1\t", round(precision[2], digits=2), "\t", round(recall[2], digits=2), "\t", classSize[2], "\n")
    println("avg\t", round((precision[1] + precision[2])/2, digits=2), "\t", round((recall[1] + recall[2])/2, digits=2))
    println("w. avg\t", round(precision[1] * classSize[1] / size(dataSet, 1) + precision[2] * classSize[2] / size(dataSet, 1), digits = 2), "\t", round(recall[1] * classSize[1] / size(dataSet, 1) + recall[2] * classSize[2] / size(dataSet, 1), digits = 2), "\n")
    
end 
