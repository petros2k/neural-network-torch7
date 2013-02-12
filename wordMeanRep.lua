require 'neuralNet'
require 'dataReader'

vocabSize = 251
nWEUnits = 50
nHidUnits = 200
nGram = 4

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
	A:mul(0.01)
	net.W[1][{{1,vocabSize},{1,nWEUnits}}] = A:clone()
	net.W[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}] = A:clone()
	net.W[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}] = A:clone()

	-- word embedding -> hid
	net.W[2] = torch.randn(net.W[2]:size())
	net.W[2]:mul(0.01)

	-- hid -> output
	net.W[3] = torch.randn(net.W[3]:size())
	net.W[3]:mul(0.01) 

	return net
end

-- update weights
function updateWeights( net, DW , DWb, rate)
	NeuralNetwork.updateWeights(net, DW, DWb, rate)

-- word -> word embedding
	local A = net.W[1][{{1,vocabSize},{1,nWEUnits}}]:clone()
	A:add(net.W[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}])
	A:add(net.W[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}])
	A:mul(-1/3.)

	net.W[1]:fill(0)
	net.W[1][{{1,vocabSize},{1,nWEUnits}}] = A:clone()
	net.W[1][{{vocabSize+1,2*vocabSize},{nWEUnits+1,2*nWEUnits}}] = A:clone()
	net.W[1][{{2*vocabSize+1,3*vocabSize},{2*nWEUnits+1,3*nWEUnits}}] = A:clone()
end

-- training net
function oneStepGradientDescent( net, X, T , rate)
	NeuralNetwork.feedForward(net, X)
	DW,DWb = NeuralNetwork.backpropagate(net, T)
	updateWeights(net, DW, DWb, rate)
end

function gradientDescent( net, TrainData, ValidData, batchSize, nEpoch, rate )
	local nSample = TrainData.X:size()[2]

	for i = 1,nEpoch do
		for j = 1,nSample/batchSize do
			local subX = TrainData.X[{{},{(j-1)*batchSize+1,j*batchSize}}]
			local subT = TrainData.T[{{},{(j-1)*batchSize+1,j*batchSize}}]
			oneStepGradientDescent( net, subX, subT, rate )
		end
		
		if math.mod(i,10) == 0 then
			NeuralNetwork.feedForward(net,ValidData.X)
			print(net.costF.apply(net.Y[net.nLayer],ValidData.T) / ValidData.X:size()[2])
		end
	end
end

-- main
function main()
	local TrainData = DataReader.loadCompactData("data/data.valid")
	--local ValidData = DataReader.loadCompactData("data/data.xxx")
	local net = createNeuralNet()
	gradientDescent( net, TrainData, TrainData, 100, 100, 0.1 )
end

main()