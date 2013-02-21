require 'mlp'

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
	beta = 0.001,
	rho = 0.05,
		
	apply = function ( Y, T , net)
		-- squared lost
		local D = torch.mul(Y, -1)
		D:add(T)
		local lost = 0.5 * D:pow(2):sum() / T:size()[2]
		
		-- regularity
		local reg = 0
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
				reg = reg + R1:sum() + R2:sum()
			end
		end

		return lost + beta*reg
	end,

	derivativeZ = function ( Y, T, Z, f)
		local dZ = f.derivative(Y, Z)
		local negT = torch.mul(T, -1)
		dZ:cmul( torch.add(Y, negT) )
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

--****************************** feedforward ******************************--
function sAutoEncoder:feedforward(X)
	mlp.feedforward(self)

	-- update Rho
	for i = 1,self.nLayer do
		if self.Rho ~= nil then
			self.Rho[i] = torch.mean(self.Y[i],2)
		end
	end
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


function test()
	local struct = { {size=100,f=nil,bias=0} , {size=10,f=AtvFunc.logistic,bias=1} , {size=100,f=AtvFunc.logistic,bias=1} }
	local net = sAutoEncoder:new(struct, CostFunc.squaredCost)
	net:print()

end

test()
