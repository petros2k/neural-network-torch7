require 'neuralNet'
require 'dataReader'

vocabSize = 251
nWEUnits = 50
nHidUnits = 200
nGram = 4

momentum = 0.9
stdInitW = 0.01

DeltaW = {}
DeltaWb = {}

torch.setdefaulttensortype('torch.FloatTensor')

-- create net
function createNeuralNet()
	local struct = {
		{	size = vocabSize * (nGram-1),
			f = nil,
			bias = 0 },
		{	size = nWEUnits * (nGram-1),
			f = NeuralNetwork.logistic,
			bias = 0 },
		{	size = nHidUnits,
			f = NeuralNetwork.logistic,
			bias = 1 },
		{	size = vocabSize,
			f = NeuralNetwork.normExp,
			bias = 1}
	}

	net = NeuralNetwork.createNeuralNet( struct, NeuralNetwork.crossEntropyCost )

-- init weights
	-- words -> word embeding
	local A = torch.randn(vocabSize, nWEUnits)
	A:mul(stdInitW)
	net.W[1][{{1,vocabSize},{1,nWEUnits}}] = A
	net.W[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}] = A
	net.W[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}] = A

	-- word embedding -> hid
	net.W[2] = torch.randn(net.W[2]:size())
	net.W[2]:mul(stdInitW)

	-- hid -> output
	net.W[3] = torch.randn(net.W[3]:size())
	net.W[3]:mul(stdInitW) 

	return net
end

-- update weights
function updateWeights( net, DW , DWb, rate)
	--NeuralNetwork.updateWeights(net, DW, DWb, rate)

-- word -> word embedding
	local A = DW[1][{{1,vocabSize},{1,nWEUnits}}]:clone()
	A:add(DW[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}])
	A:add(DW[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}])

	DW[1]:fill(0)
	DW[1][{{1,vocabSize},{1,nWEUnits}}] = A
	DW[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}] = A
	DW[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}] = A

	for i = 1,net.nLayer-1 do
		if DeltaW[i] == nil then
			DeltaW[i] = torch.Tensor(DW[i]:size()):fill(0)
		end
		DeltaW[i]:mul(momentum)
		DeltaW[i]:add(DW[i])		
		local D = torch.mul(DeltaW[i],-rate)
		net.W[i]:add(D)
	end

	for i = 1,net.nLayer do
		if net.Wb[i] ~= nil then
			if DeltaWb[i] == nil then
				DeltaWb[i] = torch.Tensor(DWb[i]:size()):fill(0)
			end
			DeltaWb[i]:mul(momentum)
			DeltaWb[i]:add(DWb[i])		
			local D = torch.mul(DeltaWb[i],-rate)
			net.Wb[i]:add(D)
		end
	end


end

-- training net
function oneStepGradientDescent( net, X, T , rate)
	NeuralNetwork.feedForward(net, X)
	local DW,DWb = NeuralNetwork.backpropagate(net, T)
	updateWeights(net, DW, DWb, rate)
end

function gradientDescent( net, TrainData, ValidData, batchSize, nEpoch, rate )
	local nSample = TrainData.nSample
	local VData = unpackData(ValidData, 1, ValidData.nSample)

	for i = 1,nEpoch do
		if math.mod(i,10) == 0 then
			NeuralNetwork.feedForward(net,VData.X)
			print(net.costF.apply(net.Y[net.nLayer],VData.T) / VData.T:size()[2])
			collectgarbage()
		end

		for j = 1,nSample/batchSize do
			local fragData = unpackData(TrainData, (j-1)*batchSize+1, j*batchSize)			
			oneStepGradientDescent( net, fragData.X, fragData.T, rate )
			if math.mod(j,300) == 0 then
				print('collect garbage')
				collectgarbage()
			end
		end		

		collectgarbage()
	end
end

function unpackData( Data, iStart, iEnd) 
	local fragData = {}
	fragData.X = torch.Tensor(Data.nInDim, iEnd-iStart+1):fill(0)
	fragData.T = torch.Tensor(Data.nOutDim, iEnd-iStart+1):fill(0)

	for i = iStart,iEnd do
		local X = Data.X[i]
		for idx,value in pairs(X) do
			fragData.X[{idx,i-iStart+1}] = value
		end

		local T = Data.T[i]
		for idx,value in pairs(T) do
			fragData.T[{idx,i-iStart+1}] = value
		end
	end

	return fragData
end

-- main
function main()
	print("load data...")
	local TrainData = DataReader.loadCompactData("data/data.train")
	local ValidData = DataReader.loadCompactData("data/data.valid")
	local net = createNeuralNet()

	print("training...")
	gradientDescent( net, TrainData, ValidData, 100, 10000, 0.1 )
end

main()
