require 'debugger'
-- restricted Boltzmann machine
rbm = {}

--************ construction ************--
function rbm:new( nHidUnits, nVisUnits )
	local net = {}
	net.W = torch.Tensor(nHidUnits, nVisUnits):fill(0)
	net.Wb = {	hid = torch.Tensor(nHidUnits,1),
				vis = torch.Tensor(nVisUnits,1)}
	
	setmetatable(net, self)
	self.__index = self
	return net
end

function rbm:initWeights() 
	self.W = torch.randn(self.W:size())
	self.W:mul(0.001)
	self.Wb.hid = torch.randn(self.Wb.hid:size())
	self.Wb.vis = torch.randn(self.Wb.vis:size())
	self.Wb.hid:mul(0.1)
	self.Wb.vis:mul(0.1)
end

--**************** cal conditional probabilities ****************--
-- Pr(hi=1 | v) = 1 / (1 + exp(- sum(wij * vj) - bi)
function rbm:vis2hidProb( V )
	local nSamples = V:size()[2]
	local E = torch.mm(self.W, V)
	E:add(torch.mm(self.Wb.hid, torch.Tensor(1,nSamples):fill(1)))
	E:mul(-1)
	E:exp()
	E:add(1)
	return torch.cdiv(torch.Tensor(E:size()):fill(1), E)
end

-- Pr(vj=1 | h) = 1 / (1 + exp(- sum(wij * hi) - cj)
function rbm:hid2visProb( H )
	local nSamples = H:size()[2]
	local E = torch.mm(self.W:t(), H)
	E:add(torch.mm(self.Wb.vis, torch.Tensor(1,nSamples):fill(1)))
	E:mul(-1)
	E:exp()
	E:add(1)
	return torch.cdiv(torch.Tensor(E:size()):fill(1), E)
end

--**************** cal goodness ********************--
-- G(h,v) = sum(hi * wij * vj) + sum(hi*bi) + sum(vj*cj)
function rbm:goodness( H, V)
	local nSamples = H:size()[2]
	local Ghv = torch.mm(self.W, V)
	Ghv:cmul(H)
	local Gh = torch.reshape(self.Wb.hid, 1, nSamples)
	Gh:cmul(H)
	local Gv = torch.reshape(self.Wb.vis, 1, nSamples)
	Gv:cmul(V)
	return (Ghv:sum() + Gh:sum() + Gv:sum()) / nSamples
end

-- dG/dwij = hivj
function rbm:goodnessDerivative( H, V)
	local nSamples = H:size()[2]
	local DW = torch.mm(H, V:t())
	DW:mul(1/nSamples)
	local DWb = {}
	DWb.hid = H:sum(2)
	DWb.hid:mul(1/nSamples)
	DWb.vis = V:sum(2)
	DWb.vis:mul(1/nSamples)

	return DW , DWb
end

--****************** Contrastive Divergence ****************--
function rbm:cd1(V0)
	local bernoulli = function (x) return 1 end
	local H0 = self:bernoulli(self:vis2hidProb(V0))
	local DW0, DWb0 = self:goodnessDerivative(H0, V0)

	local V1 = self:bernoulli(self:hid2visProb(H0))
	local H1 = self:bernoulli(self:vis2hidProb(V1))
	local DW1, DWb1 = self:goodnessDerivative(H1, V1)

	return DW0 - DW1, {hid = DWb0.hid - DWb1.hid, vis = DWb0.vis - DWb1.vis}
end

--***************** learning *********************--
function rbm:updateWeights(DeltaW, DeltaWb, DW, DWb, params)
	DeltaW:mul(params.momentum)
	DeltaW:add(DW)
	self.W:add(torch.mul(DeltaW, params.learnRate))

	DeltaWb.hid:mul(params.momentum)
	DeltaWb.hid:add(torch.mul(DWb.hid, -1))
	self.Wb.hid:add(torch.mul(DeltaWb.hid, params.learnRate))

	DeltaWb.vis:mul(params.momentum)
	DeltaWb.vis:add(torch.mul(DWb.vis, -1))
	self.Wb.vis:add(torch.mul(DeltaWb.vis, params.learnRate))
end

--********************* sampling ********************--
function rbm:bernoulli( Prob )
	local R = torch.rand(Prob:size())
	return torch.lt(R, Prob):float()
end

--require 'debugger'

function rbm:gradientDescent(Data, params)
	local nSample = Data:size()[2]

	-- for momentum updating weights
	local DeltaW = torch.Tensor(self.W:size())
	local DeltaWb = {hid = torch.Tensor(self.Wb.hid:size()), vis = torch.Tensor(self.Wb.vis:size())}

	-- run nEpoch epoches
	for i = 1, params.nEpoch do
		print(i)
		--local I = self.W[{1,{}}]:clone():resize(28,28)
		--image.display(I)
		--print(I)

		for i, = 1,nSample/params.batchSize do
			local V = self:bernoulli(Data[{{},{(j-1)*params.batchSize+1, j*params.batchSize}}]:clone())
			local DW, DWb = self:cd1(V)
			self:updateWeights( DeltaW, DeltaWb, DW, DWb, params )		
			collectgarbage()
		end
	end	
end




