require 'rbm'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

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
	Data.X:mul(1. / 255)
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

-- main --
function main()

	local params = {}

	params.momentum = 0.9
	params.learnRate = 0.1
	params.nEpoch = 5
	params.batchSize = 100
	params.nHidUnits = 300

	print('load data...')
	--local Data = loadData("digitData/train-images.idx3-ubyte", "digitData/train-labels.idx1-ubyte").X
	local Data = loadData("digitData/t10k-images.idx3-ubyte", "digitData/t10k-labels.idx1-ubyte").X
	params.nVisUnits = Data:size(1)

	local net = rbm:new(params.nHidUnits, params.nVisUnits)
	net:initWeights()
	
	print('training network...')

	timer = torch.Timer()
	net:gradientDescent (Data, params)
	print('timer:', timer:time().real)

	-- visualize
	for i = 1,50 do
		local I = net.W[{i,{}}]:clone():resize(28,28)
		image.display(I)
	end
end

main()


