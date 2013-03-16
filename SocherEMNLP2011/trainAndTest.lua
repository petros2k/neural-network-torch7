require 'RAE'
require 'optim'

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
			sum = sum + 1
			if sum > 500 then break end
		end
		file:close()
	end

	local dicLen = nextWID - 1
	return Sentences, Labels, wordMap, dicLen
end

--********************** train *********************


--*********************** main  ********************
--require 'debugger'

function main()

	-- load data
	print('load data...')
	local Sentences, Labels, wordMap, dicLen = loadData( 'data/rt-polarity/sentences_neg.txt', 'data/rt-polarity/sentences_pos.txt')
		
	-- create net
	local dim = 100
	local nCat = 2

	print('create network...')
	local L = torch.randn(dim, dicLen):mul(0.01)
	local struct = {nCategory = 2, Lookup = L , func = tanh, funcPrime = norm2TanhPrime }
	local rae = reAutoEncoder:new(struct)

	print('train...')
	rae:train(Sentences, Labels, 100, optim.lbfgs, {maxIter=50, learningRate=0.1})
end

main()
