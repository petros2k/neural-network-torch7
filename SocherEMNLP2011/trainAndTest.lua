require 'RAE'
require 'optim'

--*********************************** test with toy  ******************************--
function test ()
        local rae = reAutoEncoder:load( 'rae.txt' )
        local Sentences = {
                        torch.Tensor({2,3}),
                        torch.Tensor({1,3,4}),
                        torch.Tensor({1,2,3,5,6}),
                        torch.Tensor({2})
                }
        local Labels = {
                        torch.Tensor({1,0}):resize(2,1),
                        torch.Tensor({0,1}):resize(2,1),
                        torch.Tensor({1,0}):resize(2,1),
                        torch.Tensor({1,0}):resize(2,1)
                }


        config = {lambda = 1e-4, alpha = 0.2, nProcess = 1}
        --print( rae:checkGradient( {Sentences[1],Sentences[2]}, {Labels[1],Labels[1]} , config)  ) 
        --print( rae:checkGradient( {Sentences[3]}, Labels , config)  ) 
        --print( rae:checkGradient( {Sentences[4]}, Labels , config)  ) 
        print(rae:checkGradient(Sentences, Labels, config))
end

--test()

--********************** load data **********************--
--also refine word indices (and store the map in wordMap)
function loadData( negPath, posPath )
	local Sentences = {}
	local Labels = {}
	
	local split = function(s)
		local t = {}
		local i = 1
		for k in string.gmatch(s, "[^%s]+") do
			t[i] = k
			i = i + 1
		end
		return t
	end
	
	local i = 1
	local paths = {posPath, negPath}

	local wordMap = {}
	local nextWID = 1

	for j = 1,2 do
		local file = torch.DiskFile(paths[j], 'r')
		file:quiet()
		local sum = 0
		while true do 
			local str = file:readString('*l')
			if file:hasError() then break end
			Sentences[i] = torch.Tensor(split(str))
			Labels[i] = torch.zeros(#paths,1)
			Labels[i][{j}] = 1

			for k = 1,Sentences[i]:size(1) do
				local t = wordMap[Sentences[i][k]]
				if t == nil then 
					t = nextWID
					wordMap[Sentences[i][k]] = t
					nextWID = nextWID + 1
				end
				Sentences[i][{k}] = t
			end
			i = i + 1
			
			-- for test only
			--sum = sum + 1
			--if sum > 100 then break end
		end
		file:close()
	end

	local dicLen = nextWID - 1
	return Sentences, Labels, wordMap, dicLen
end

function splitData (Sentences , Labels , nPart)
	local trainSen = {}
	local trainLab = {}
	local testSen = {}
	local testLab = {}

	local nSen = #Sentences
	local trainId = 1
	local testId = 1
	for i = 1, nSen do
		if math.mod(i,nPart) == 0 then
			testSen[testId] = Sentences[i]
			testLab[testId] = Labels[i]
			testId = testId + 1
		
		else
			trainSen[trainId] = Sentences[i]
			trainLab[trainId] = Labels[i]
			trainId = trainId + 1
		end
	end

	return {
		train = {Sentences = trainSen, Labels = trainLab}, 
		test = {Sentences = testSen, Labels = testLab } 
	}
end

--*********************** main  ********************
--require 'debugger'
require 'mlp'

local MODE_TRAIN = 1
local MODE_TEST = 2

function main()
	
	local mode = MODE_TEST	

	-- load data
	print('load data...')
	local Sentences, Labels, wordMap, dicLen = loadData( 'data/rt-polarity/sentences_neg.txt', 'data/rt-polarity/sentences_pos.txt')
	local Data = splitData(Sentences, Labels, 5)
	
	if mode == MODE_TRAIN then

		-- create net
		local dim = 100
		local nCat = 2

		print('create network...')
		local L = torch.randn(dim, dicLen):mul(0.01)
		local struct = {nCategory = 2, Lookup = L , func = tanh, funcPrime = norm2TanhPrime }
		local rae = reAutoEncoder:new(struct)

		print('train...')
		rae:train(Data, 100, optim.lbfgs, 
				{maxIter=100, learningRate=1},
				{alpha = 0.2, lambda = 1e-4})

	elseif mode == MODE_TEST then

		rae = reAutoEncoder:load('model.7')
		local classifier = rae:trainFinalClassifier(Data.train.Sentences, Data.train.Labels, 
				optim.lbfgs, {maxIter=1000, learningRate=1})
			
		local accuracy = rae:test(classifier, Data.test.Sentences, Data.test.Labels)
		print('accuracy: ' .. accuracy) io.flush()
	end
end

main()
