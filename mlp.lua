-- Multilayer Perceptron

mlp = {}
mlp_mt = {__index = mlp}


function mlp:print()
	print('mlp')
end

--************************* construction ***********************--
-- create a neural network
-- struct = { {size, f} }
--   size: number of nodes, 
--   f: activation function

function mlp:new( struct , costF )
	local net = {}
	net.Y = {} -- self.Y[layer_id] : output of units in layer_id
	net.W = {} -- self.W[layer_id] : weights of links from layer_id to layer_id+1
	net.Wb = {} -- self.Wb[layer_id] : weights of links for bias for units in layer_id
	net.F = {} -- self.F[layer_id] : every node in the same layer uses the same activation function

	if struct == nil then
		return net
	end

	net.costF = costF
	net.nLayer = #struct

	for i,s in ipairs(struct) do
		net.F[i] = s.f
		if struct[i].bias == 1 then
			net.Wb[i] = torch.Tensor(1, s.size):fill(0)
		end
	end

	for i = 1,net.nLayer-1 do
		net.W[i] = torch.Tensor(struct[i].size, struct[i+1].size):fill(0)
	end

	setmetatable(net, mlp_mt)
	return net
end

function mlp:initWeights() 
	for i = 1,self.nLayers-1 do
		self.W[i] = torch.randn(self.W[i]:size())
		self.W[i]:mul(0.01)
	end
	for i = 1,self.nLayer-1 do
		if self.Wb[i] ~= nil then
			self.Wb[i] = torch.randn(self.Wb[i]:size())
			self.Wb[i]:mul(0.01)
		end
	end
end

--********************** feed forward *************************--
--   net : network created by the function createNeuralNet
--   X = [X1 X2 ... Xn] input
function mlp:feedforward( X )
	local batchSize = X:size()[2]

-- input
	self.Y[1] = X:clone()

-- other layers
	for j = 2,self.nLayer do
		local Z = torch.mm(self.W[j-1]:t(), self.Y[j-1])
		-- if using bias
		if self.Wb[j] ~= nil then
			local T = torch.mm(self.Wb[j]:t(),torch.Tensor(1,batchSize):fill(1))
			Z:add(T)
		end
		self.Y[j] = self.F[j].apply(Z)
	end

	return self.Y[self.nLayer]
end

--***************************** backpropagate ******************************--
--   T : goldstandard
function mlp:backpropagate( T )
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
		DZ[i]:cmul(dYdZ)
	end

-- calculate DW
	for i = self.nLayer-1,1,-1 do
		DW[i] = torch.mm(self.Y[i], DZ[i+1]:t())
	end
	
	for i = self.nLayer,1, -1 do
		if self.Wb[i] ~= nil then
			Yb = torch.Tensor(1,batchSize):fill(1)
			DWb[i] = torch.mm(Yb, DZ[i]:t())
		end
	end

	return DW, DWb
end

--***************** update weight ***************--
-- newW = oldW + rate * DeltaW
function mlp:updateWeights( DeltaW , DeltaWb, rate)
	for i = 1,self.nLayer-1 do
		self.W[i]:add(torch.mul(DeltaW[i],rate))
	end
	for i = 1,self.nLayer do
		if self.Wb[i] ~= nil then
			self.Wb[i]:add(torch.mul(DeltaWb[i],rate))
		end
	end
end

--******************************* train networks with gradient descent method *************************
function mlp:gradientDescent( X, T, batchSize, nEpoch, rate )
	local nSample = X:size()[2]

	for i = 1,nEpoch do
		for j = 1,nSample/batchSize do
			local subX = X[{{},{(j-1)*batchSize+1,j*batchSize}}]
			local subT = T[{{},{(j-1)*batchSize+1,j*batchSize}}]
			self:feedForward( subX)
			DW,DWb = self:backpropagate( subT)
			self:updateWeights( DW, DWb, rate)
		end
		
		Y = self:feedForward(X)
		print(self.costF.apply(Y,T,net))

		collectgarbage()
	end
end

--***************************** gradient check **************************--
-- check if we correctly calculate gradients for W[:][1,1] and Wb[:][1]
function mlp:checkGradient( X , T )
	local Y = self:feedforward(X)
	local DW, DWb = self:backpropagate(T)
	local eps = 0.00001
	local theta = 0.00001
	local good = true

	for i = 1,self.nLayer-1 do
		self.W[i][{{1},{1}}]:add(eps)
		local rPlus = self.costF.apply(self:feedforward(X),T,net)
		self.W[i][{{1},{1}}]:add(-2*eps)
		local rMinus = self.costF.apply(self:feedforward(X),T,net)
		self.W[i][{{1},{1}}]:add(eps)

		--print((rPlus - rMinus) / (2*eps))
		--print(DW[i][{1,1}])

		if math.abs((rPlus - rMinus) / (2*eps) - DW[i][{1,1}]) > theta then 
			good = false
			break
		end
	end

	if good == true then
		for i = 1,self.nLayer do
			if self.Wb[i] ~= nil then
				self.Wb[i][{{1},{1}}]:add(eps)
				local rPlus = self.costF.apply(self:feedforward(X),T,net)
				self.Wb[i][{{1},{1}}]:add(-2*eps)
				local rMinus = self.costF.apply(self:feedforward(X),T,net)
				self.Wb[i][{{1},{1}}]:add(eps)

				--print((rPlus - rMinus) / (2*eps))
				--print(DWb[i][{1,1}])

				if math.abs((rPlus - rMinus) / (2*eps) - DWb[i][{1,1}]) > theta then 
					good = false
					break
				end
			end
		end
	end

	return good
end


--************************ activation function **********************--
AtvFunc = {}
-- Y = 1 / (1 + exp(-Z))
AtvFunc.logistic = {
	apply = function ( Z )
		local X = torch.mul(Z,-1)
		X:exp()
		X:add(1)
		local Y = torch.Tensor(X:size()):fill(1)
		Y:cdiv(X)
		return Y
	end,

	derivative = function ( Y, Z )
		local dZ = torch.mul(Y,-1)
		dZ:add(1)
		dZ:cmul(Y)
		return dZ
	end 
}

-- Y = Z
AtvFunc.indentity = {
	apply = function (Z)
		return Z
	end,
	derivative = function ( Y, Z)
		return torch.Tensor(Y:size()):fill(1)
	end
}

-- Y = exp(Z) / normalize_factor
AtvFunc.normExp = {
	apply = function (Z)
		local Y = torch.exp(Z)
		local N = Y:sum(1)
		N = torch.mm(torch.Tensor(Y:size()[1],1):fill(1), N)
		Y:cdiv(N)
		return Y
	end,
	-- do not use
	derivative = function (Y, Z)
		return nill
	end
}


--************************ cost **********************--
CostFunc = {}
-- E = 0.5 * sum ( T[i] - Y[i])^2
CostFunc.squaredCost = {
	apply = function ( Y, T , net)
		local D = torch.mul(Y, -1)
		D:add(T)
		return 0.5 * D:pow(2):sum() / T:size()[2]
	end,
	-- Y = f(Z)
	derivativeZ = function ( Y, T, Z, f)
		local dZ = f.derivative(Y, Z)
		local negT = torch.mul(T, -1)
		dZ:cmul( torch.add(Y, negT) )
		dZ:mul(1 / T:size()[2])
		return dZ
	end
}

-- E = - sum { T[i] * logY[i] }
CostFunc.crossEntropyCost = {
	apply = function( Y, T , net)
		local D = torch.log(Y)
		--if _DEBUG_ then pause() end
		D:cmul(T)
		return -1 * D:sum() / T:size()[2]
	end,
	-- Y = f(Z)
	-- f has to be normExp
	derivativeZ = function( Y, T, Z, f)
		local negT = torch.mul(T, -1)
		negT:add(Y)
		negT:mul(1 / T:size()[2])
		return negT
	end
}

--************************* for testing **************************--
function test1()
-- first network: 
	net = mlp:new( { {size=2,f=nil,bias=0} , {size=2,f=AtvFunc.logistic,bias=1} , {size=1,f=AtvFunc.logistic,bias=1} } , CostFunc.squaredCost)

	net.W[1] = torch.Tensor( { {0.1,0.2},{0.3,0.4} } )
	net.W[2] = torch.Tensor( { {0.2},{0.4} } )
	net.Wb[2] = torch.Tensor( {0.1,0.1} ):resize(1,2)
	net.Wb[3] = torch.Tensor( {0.3} ):resize(1,1)

	X = torch.Tensor({{3,2},{3,2}}):t()
	T = torch.Tensor({1,1}):resize(1,2)
	Y = net:feedforward(X)
	print( Y )
	DW, DWb = net:backpropagate(T)
	print( DW[2] )

	print(net:checkGradient(X, T))

-- second network: with softmax
	net = mlp:new( { {size=2,f=nil,bias=0} , {size=2,f=AtvFunc.logistic,bias=1} , {size=2,f=AtvFunc.normExp,bias=1} } , CostFunc.crossEntropyCost)

	net.W[1] = torch.Tensor( { {0.1,0.2},{0.2,0.3} } )
	net.W[2] = torch.Tensor( { {0.2,0.1},{0.1,0.2} } )
	net.Wb[2] = torch.Tensor( {0.1,0.2} ):resize(1,2)
	net.Wb[3] = torch.Tensor( {0.3,0.4} ):resize(1,2)

	X = torch.Tensor({{0.2,0.7},{0.2,0.7}}):t()
	T = torch.Tensor({{0,0},{1,1}}):resize(2,2)
	Y = net:feedforward(X)
	--print(self.Y[self.nLayer])
	DW, DWb = net:backpropagate(T)
	print( DWb[2] )
end

function test2()
	DataReader.createData("data.txt")
	Data = DataReader.loadCompleteData("data.txt")

	net = NeuralNetwork.createNeuralNet( { {size=500,f=nil,bias=0} , {size=200,f=NeuralNetwork.logistic,bias=1} , {size=2,f=NeuralNetwork.normExp,bias=1} } , NeuralNetwork.crossEntropyCost)

	--self.W[1] = torch.Tensor( { {0.1,0.2},{0.2,0.3} } )
	--self.W[2] = torch.Tensor( { {0.2,0.1},{0.1,0.2} } )
	--self.Wb[2] = torch.Tensor( {0.1,0.2} ):resize(1,2)
	--self.Wb[3] = torch.Tensor( {0.3,0.4} ):resize(1,2)

	NeuralNetwork.gradientDescent( net, Data.X, Data.T, 100, 1000000, 0.1)
	
end

--*********************************************************************

--test1()

