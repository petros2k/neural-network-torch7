require 'mlp'
require 'image'
require 'optim'

-- sAutoEncoder is a subclass of mlp
sAutoEncoder = {}
sAutoEncoder_mt = {__index = sAutoEncoder}
setmetatable(sAutoEncoder, {__index = mlp} )

function sAutoEncoder:print()
	mlp.print(self)
	print('sparse auto encoder')
end

--************************** cost function **********************--
CostFunction = {
	beta = 3,
	rho = 0.01,
	lambda = 0.0001,
		
	apply = function ( Y, T , net)
		-- squared lost
		local D = torch.mul(Y, -1)
		D:add(T)
		local lost = 0.5 * D:pow(2):sum() / T:size()[2]

		-- sparse
		local reg1 = 0
		local rho = CostFunction.rho
		for i = 1,net.nLayer do
			if net.Rho[i] ~= nil then
				local R1 = torch.Tensor(net.Rho[i]:size()):fill(rho)
				R1:cdiv(net.Rho[i])
				R1:log()
				R1:mul(rho)
				local R2 = torch.Tensor(net.Rho[i]:size()):fill(1-rho)
				R2:cdiv(torch.mul(net.Rho[i],-1):add(1))
				R2:log()
				R2:mul(1-rho)				
				reg1 = reg1 + R1:sum() + R2:sum()
			end
		end
		
		-- regulation
		local reg2 = 0
		for i = 1, net.nLayer-1 do
			reg2 = reg2 + torch.cmul(net.W[i],net.W[i]):sum()
		end

		return lost + CostFunction.beta*reg1 + 0.5*CostFunction.lambda*reg2
	end,

	derivativeZ = function ( Y, T, Z, f)
		local dZ = f.derivative(Y, Z)
		local negT = torch.mul(T, -1)
		dZ:cmul( torch.add(Y, negT) )
		dZ:mul(1 / T:size()[2])
		return dZ
	end
}

--******************************** construction *************************--
function sAutoEncoder:new( struct, costFunc )
	local net = mlp:new( struct, costFunc )

	net.Rho = {}
	for i,s in ipairs(struct) do
		if s.sparse == 1 then
			net.Rho[i] = torch.Tensor(s.size,1):fill(0)
		end
	end

	setmetatable(net, sAutoEncoder_mt)
	return net
end

function sAutoEncoder:initWeights() 
	for i = 1,self.nLayer-1 do
		self.W[i] = torch.rand(self.W[i]:size())
		self.W[i]:mul(0.25):add(-0.12)
	end
	for i = 1,self.nLayer do
		if self.Wb[i] ~= nil then
			self.Wb[i] = torch.rand(self.Wb[i]:size())
			self.Wb[i]:mul(0.45):add(-0.22)
		end
	end
end

--****************************** feedforward ******************************--
function sAutoEncoder:feedforward(X)
	mlp.feedforward(self, X)

	-- update Rho
	for i = 1,self.nLayer do
		if self.Rho[i] ~= nil then
			self.Rho[i] = torch.mean(self.Y[i],2)
		end
	end
	return self.Y[self.nLayer]
end

--***************************** backpropagate ******************************--
--   T : goldstandard
function sAutoEncoder:backpropagate( T )
	local batchSize = T:size()[2]
	local DZ = {}
	local DW = {}
	local DWb = {}

-- calculate DZ
	-- for output units
	DZ[self.nLayer] = self.costF.derivativeZ( self.Y[self.nLayer], T, nil, self.F[self.nLayer] )

	-- for other layers
	for i = self.nLayer-1,2,-1 do
		local dYdZ = self.F[i].derivative( self.Y[i], nil )
		DZ[i] = torch.mm(self.W[i], DZ[i+1])
		if self.Rho[i] ~= nil then
			local R1 = torch.Tensor(DZ[i]:size()):fill(self.costF.rho)
			R1:cdiv(torch.mm(self.Rho[i], torch.Tensor(1,batchSize):fill(1)))
			R1:mul(-1)
			local R2 = torch.Tensor(DZ[i]:size()):fill(1-self.costF.rho)
			R2:cdiv(torch.mm(self.Rho[i], torch.Tensor(1,batchSize):fill(1)):mul(-1):add(1))
			R1:add(R2):mul(self.costF.beta):mul(1/batchSize)
			DZ[i]:add(R1)
		end

		DZ[i]:cmul(dYdZ)
	end

-- calculate DW
	for i = self.nLayer-1,1,-1 do
		DW[i] = torch.mm(self.Y[i], DZ[i+1]:t()) + torch.mul(self.W[i], self.costF.lambda)
	end
	
	for i = self.nLayer,1, -1 do
		if self.Wb[i] ~= nil then
			Yb = torch.Tensor(1,batchSize):fill(1)
			DWb[i] = torch.mm(Yb, DZ[i]:t())
		end
	end

	return DW, DWb
end

--*************************** visualize *************************--
-- only for 1-hidden-layer net
function sAutoEncoder:visualize( nRow, nCol )
	for i = 1,self.W[1]:size(2) do
		local I = self.W[1][{{},i}]:clone():resize(nRow,nCol)
		local s = math.sqrt(torch.cmul(I,I):sum())
		I:div(s)
		image.display({image=I,zoom=5})
		--print(I)
	end
end

--***************************************** test *****************************************--
iRow = 512
iCol = 512
pRow = 8
pCol = 8
nImages = 10
nPatch = 10000

torch.setdefaulttensortype('torch.FloatTensor')

function loadData( path )
	local f = torch.DiskFile.new(path)
	local Image = f:readFloat(iRow*iCol*nImages)
	Image = torch.Tensor(Image):resize(nImages,iRow,iCol)
	f:close()

	local X = torch.Tensor(pRow*pCol,nPatch):fill(0)
	-- sampling
	local I = torch.rand(nPatch):mul(nImages):add(1):int()
	local J = torch.rand(nPatch):mul(iRow - pRow):add(1):int()
	local K = torch.rand(nPatch):mul(iCol - pCol):add(1):int()
	for i = 1,nPatch do
		X[{{},i}] = Image[{I[i],{J[i],J[i]+pRow-1},{K[i],K[i]+pCol-1}}]
	end

	-- normalize
	X = mlp:normalizeData(X)

	return X
end

function test()
	print('load data...')
	local X = loadData('image')

	local struct = {	{size=pRow*pCol,f=nil,bias=0,sparse=0} , 
						{size=25,f=AtvFunc.logistic,bias=1,sparse=1} ,
						{size=pRow*pCol,f=AtvFunc.logistic,bias=1,sparse=0} }

	--local net = mlp:new(struct, CostFunc.squaredCost)
	local net = sAutoEncoder:new(struct, CostFunction)
	net:initWeights()
	
	print('training...')
	net:train( X, X, nPatch, 1, optim.lbfgs , {maxIter = 100, learningRate = 1})
	net:visualize( pRow, pCol )
end

test()
