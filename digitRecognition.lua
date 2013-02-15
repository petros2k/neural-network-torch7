require 'neuralNet'
require 'debugger'

torch.setdefaulttensortype('torch.FloatTensor')

-- error rate
function errorRate( Y, T )
	local temp, IY = torch.max(Y,1)
	temp, IT = torch.max(T,1)
	return torch.ne(IY,IT):sum() / Y:size()[2]
end

-- C = E + wdCoeff/2 * sum_i(wi^2)
function calLost( net, T, config )
	local lost = net.costF.apply(net.Y[net.nLayer], T) / T:size()[2]
	local wLost = 0
	for i,W in pairs(net.W) do
		wLost = wLost + torch.pow(W,2):sum()
	end
	for i,Wb in pairs(net.Wb) do
		wLost = wLost + torch.pow(Wb,2):sum()
	end
	return lost + config.wdCoeff * wLost/2
end

-- dC/dwi = dE/dwi + wdCoeff * wi
function calDLostDW( net, T, config )
	local DW, DWb = NeuralNetwork.backpropagate( net, T )
	
	for i = 1, net.nLayer-1 do
		DW[i]:add(torch.mul(net.W[i], config.wdCoeff))
	end
	for i = 1, net.nLayer do
		if net.Wb[i] ~= nil then
			DWb[i]:add(torch.mul(net.Wb[i], config.wdCoeff))
		end
	end
	
	return DW, DWb
end

-- delta_wi = delta_wi_old * momentum - dE/dwi
-- wi = wi + learnRate * delta_wi
function updateWeights( net, DW, DWb, DeltaW, DeltaWb, config )
	for i = 1, net.nLayer-1 do
		DeltaW[i]:mul(config.momentum)
		DeltaW[i]:add(torch.mul(DW[i],-1))
		net.W[i]:add(torch.mul(DeltaW[i], config.learnRate))
	end

	for i = 1, net.nLayer do
		if net.Wb[i] ~= nil then
			DeltaWb[i]:mul(config.momentum)
			DeltaWb[i]:add(torch.mul(DWb[i],-1))
			net.Wb[i]:add(torch.mul(DeltaWb[i], config.learnRate))
		end
	end
end

-- training net with gradient descent
function gradientDescent( net, TData, VData, TestData, config)
	local nSample = TData.X:size()[2]
	local trainLostGraph = {}
	local validLostGraph = {}

	local bestNet = nil
	local lowestValidLost = 100000

	-- for momentum updating weights
	local DeltaW = {}
	local DeltaWb = {}

	for i = 1, net.nLayer-1 do
		DeltaW[i] = torch.Tensor(net.W[i]:size()):fill(0)		
	end
	for i = 1, net.nLayer do
		if net.Wb[i] ~= nil then
			DeltaWb[i] = torch.Tensor(net.Wb[i]:size()):fill(0)
		end
	end

	-- run nEpoch epoches
	for i = 1, config.nEpoch do
		print('------ epoch ' .. i)
		if math.mod(i,1) == 0 then
			NeuralNetwork.feedForward(net,VData.X)
			print("validate " .. calLost(net , VData.T, config))
			print("error rate " .. errorRate( net.Y[net.nLayer], VData.T))

			NeuralNetwork.feedForward(net, TestData.X)
			print("validate " .. calLost(net , TestData.T, config))
			print("error rate " .. errorRate( net.Y[net.nLayer], TestData.T))
			collectgarbage()
		end

		for j = 1,nSample/config.batchSize do
			local T = TData.T[{{},{(j-1)*config.batchSize+1, j*config.batchSize}}]
			local X = TData.X[{{},{(j-1)*config.batchSize+1, j*config.batchSize}}]

			-- test on batch
			if math.mod(j,100) == 0 then 
				NeuralNetwork.feedForward(net,X)
				print("test batch " .. j .. " : " .. calLost(net , T, config))
				collectgarbage()
			end

			-- one step gradient descent 
			NeuralNetwork.feedForward( net, X )
			local DW, DWb = calDLostDW( net, T, config )
			updateWeights( net, DW, DWb, DeltaW, DeltaWb, config )		
		end
		collectgarbage()
	end
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
	net = NeuralNetwork.createNeuralNet( struct, NeuralNetwork.crossEntropyCost )
	
	-- init weights
	for i = 1, net.nLayer-1 do
		net.W[i] = torch.randn(net.W[i]:size())
		net.W[i]:mul(0.1)
	end

	for i = 1, net.nLayer do
		if net.Wb[i] ~= nil then
			net.Wb[i] = torch.randn(net.Wb[i]:size())
			net.Wb[i]:mul(0.1)
		end
	end
	return net
end

-- main --
function main()

	local config = {}

	local nInputUnits = 28*28
	local nOutputUnits = 10
	local nHidUnits = 500

	config.momentum = 0.9
	config.learnRate = 0.1
	config.nEpoch = 50
	config.batchSize = 100
	config.wdCoeff = 0

	local struct = {
		{	size = nInputUnits,
			f = nil,
			bias = 0 },
		{	size = nHidUnits,
			f = NeuralNetwork.logistic,
			bias = 1 },
		{	size = nOutputUnits,
			f = NeuralNetwork.normExp,
			bias = 1}
	}

	local net = createNeuralNet( struct )
	local TrainData, ValidData = splitDataForValidation( loadData("digitData/train-images.idx3-ubyte", "digitData/train-labels.idx1-ubyte"), 10 )
	local TestData = loadData("digitData/t10k-images.idx3-ubyte", "digitData/t10k-labels.idx1-ubyte")
	gradientDescent ( net, TrainData, ValidData, TestData, config)
	
end

main()


