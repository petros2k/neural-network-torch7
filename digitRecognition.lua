require 'mlp'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

-- error rate
function errorRate( Y, T )
	local temp, IY = torch.max(Y,1)
	temp, IT = torch.max(T,1)
	return torch.ne(IY,IT):sum() / Y:size()[2]
end

-- read data
function loadData( imagePath, labelPath ) 
	local Data = {}

	-- read images
	local file = torch.DiskFile.new(imagePath)
	file:binary()
	file:bigEndianEncoding()

	local buff = file:readInt(4):resize(4)
	local nSample = buff[2]
	local nRow = buff[3]
	local nCol = buff[4]

	Data.X = torch.ByteTensor(file:readByte(nSample*nRow*nCol)):resize(nSample,nRow*nCol):float():t()
	-- normalize
	Data.X:mul(2. / 255)
	Data.X:add(-1)
	file:close()

	-- read labels
	file = torch.DiskFile.new(labelPath)
	file:binary()
	file:bigEndianEncoding()

	buff = file:readInt(2):resize(2)
	buff = file:readByte(nSample):resize(nSample)
	
	Data.T = torch.Tensor(10, nSample):fill(0)
	for i = 1,nSample do
		if buff[i] == 0 then buff[i] = 10; end
		Data.T[{buff[i],i}] = 1
	end
	
	file:close()

	return Data
end

-- split valid data
function splitDataForValidation( Data , nPart)
	local TrainData = {}
	local ValidData = {}
	local nTrain = (nPart - 1.) / nPart * Data.X:size()[2]

	TrainData.X = Data.X[{{},{1,nTrain}}]
	TrainData.T = Data.T[{{},{1,nTrain}}]

	ValidData.X = Data.X[{{},{nTrain+1, Data.X:size()[2]}}]
	ValidData.T = Data.T[{{},{nTrain+1, Data.X:size()[2]}}]

	return TrainData, ValidData
end

-- create net
function createNeuralNet( struct )
	local net = mlp:new( struct, CostFunc.crossEntropyCost )
	net:initWeights()
	return net
end

-- main --
function main()

	local config = {}

	local nInputUnits = 28*28
	local nOutputUnits = 10
	local nHidUnits = 500

	local struct = {
		{	size = nInputUnits,
			f = nil,
			bias = 0 },
		{	size = nHidUnits,
			f = AtvFunc.logistic,
			bias = 1 },
		{	size = nOutputUnits,
			f = AtvFunc.normExp,
			bias = 1}
	}

	local net = createNeuralNet( struct )

	-- load data
	print('load train data')
	local TrainData, ValidData = splitDataForValidation( loadData("digitData/train-images.idx3-ubyte", "digitData/train-labels.idx1-ubyte"), 10 )
	print('load test data')
	local TestData = loadData("digitData/t10k-images.idx3-ubyte", "digitData/t10k-labels.idx1-ubyte")
	
	-- train
	print('training...')
	net:train( TrainData.X, TrainData.T, 5000, optim.lbfgs , {maxIter = 200, learningRate = 1})
	
	-- test
	local Y = net:feedforward(TestData.X)
	local err = errorRate(Y, TestData.T)
	print(err)
	
end

main()


