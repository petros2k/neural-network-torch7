--************************** Neural Network *********************--

-- create a neural network
-- struct = { {size, f} }
--   size: number of nodes, 
--   f: activation function
function createNeuralNet( struct , batchSize , costF)
    local net = {}
    net.Y = {} -- net.Y[layer_id] : output of units in layer_id
    net.W = {} -- net.W[layer_id] : weights of links from layer_id to layer_id+1
    net.Wb = {} -- net.Wb[layer_id] : weights of links for bias for units in layer_id
    net.F = {} -- net.F[layer_id] : every node in the same layer uses the same activation function
    net.batchSize = batchSize
    net.costF = costF

    net.nLayer = table.getn(struct)

    for i,s in ipairs(struct) do
       net.Y[i] = torch.Tensor(s.size, batchSize):fill(0)
       net.F[i] = s.f
       if struct[i].bias == 1 then
          net.Wb[i] = torch.Tensor(1, s.size):fill(0)
       end
    end

    for i = 1,net.nLayer-1 do
       net.W[i] = torch.Tensor(struct[i].size + 1, struct[i+1].size + 1):fill(0)
    end

    return net
end



-- feed forward
--   net : network created by the function createNeuralNet
--   X = [X1 X2 ... Xn] input
function feedForward( net , X )
-- input
    net.Y[1] = X:clone()

-- other layers
    for j = 2,net.nLayer do
       local Wt = net.W[j-1]:t()
       local Z = torch.mm(Wt, net.Y[j-1])
       -- if using bias
       if net.Wb[j] ~= nil then
          local T = torch.mm(net.Wb[j]:t(),torch.Tensor(1,net.batchSize):fill(1))
          Z:add(T)   
       end
       net.Y[j] = net.F[j].apply(Z)
    end
end

-- backpropagate
--   net : network created by the function createNeuralNet
--   T : golden
function backpropagate( net, T )
    local DZ = {}
    local DW = {}
    local DWb = {}

-- calculate DZ
    -- for output units
    DZ[net.nLayer] = net.costF.derivativeZ( net.Y[net.nLayer], T, nil, net.F[net.nLayer] )

    -- for other layers
    for i = net.nLayer-1,2,-1 do
       local dYdZ = net.F[i].derivative( net.Y[i], nill )
       DZ[i] = torch.mm(net.W[i], DZ[i+1])
       DZ[i]:cmul(dYdZ)
    end


-- calculate DW
    for i = net.nLayer-1,1,-1 do
       DW[i] = torch.mm(net.Y[i], DZ[i+1]:t())
       DW[i]:mul(1. / net.batchSize)
    end
    
    for i = net.nLayer,1, -1 do
       if net.Wb[i] ~= nil then
          Yb = torch.Tensor(1,net.batchSize):fill(1)
          DWb[i] = torch.mm(Yb, DZ[i]:t())
          DWb[i]:mul(1. / net.batchSize)
       end
    end

    return DW, DWb
end

-- update weight for a neural network
function updateWeights( net, DW , DWb, rate)
    for i = 1,net.nLayer do
       DW[i]:mul(rate)
       net.W[i]:add(DW[i])
       if net.Wb[i] ~= nil then
          DWb[i]:mul(rate)
          net.Wb[i]:add(DWb)
       end
    end
end

--************************ activation function **********************--

-- Y = 1 / (1 + exp(-Z))
logistic = {
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
indentity = {
    apply = function (Z)
       return Z
    end,
    derivative = function ( Y, Z)
       return torch.Tensor(Y:size()):fill(1)
    end
}

-- Y = exp(Z) / normalize_factor
normExp = {
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

-- E = 0.5 * sum ( T[i] - Y[i])^2
squaredCost = {
    apply = function ( Y, T )
       local D = torch.mul(Y, -1)
       D:add(T)
       return 0.5 * D:pow(2):sum()
    end,
    -- Y = f(Z)
    derivativeZ = function ( Y, T, Z, f)
       local dZ = f.derivative(Y, Z)
       dZ:cmul( torch.add(Y, negT) )
       return dZ
    end
}

-- E = - sum { T[i] * logY[i] }
crossEntropyCost = {
    apply = function( Y, T)
       local D = torch.log(Y)
       D:cmul(T)
       return -1 * D:sum()
    end,
    -- Y = f(Z)
    -- f has to be normExp
    derivativeZ = function( Y, T, Z, f)
       local negT = torch.mul(T, -1)
       negT:add(Y)
       return negT
    end
}

--************************* main **************************--
--[[
net = createNeuralNet( { {size=2,f=nil,bias=0} , {size=2,f=logistic,bias=1} , {size=1,f=logistic,bias=1} } , 2 , squaredCost)

net.W[1] = torch.Tensor( { {0.1,0.2},{0.3,0.4} } )
net.W[2] = torch.Tensor( { {0.2},{0.4} } )
net.Wb[2] = torch.Tensor( {0.1,0.1} ):resize(1,2)
net.Wb[3] = torch.Tensor( {0.3} ):resize(1,1)

X = torch.Tensor({{3,2},{3,2}}):t()
T = torch.Tensor({1,1}):resize(1,2)
feedForward(net, X)
print(net.Y[net.nLayer])
DW, DWb = backpropagate(net, T)
print( DW[2] )
]]--

net = createNeuralNet( { {size=2,f=nil,bias=0} , {size=2,f=logistic,bias=1} , {size=2,f=normExp,bias=1} } , 2 , crossEntropyCost)

net.W[1] = torch.Tensor( { {0.1,0.2},{0.2,0.3} } )
net.W[2] = torch.Tensor( { {0.2,0.1},{0.1,0.2} } )
net.Wb[2] = torch.Tensor( {0.1,0.2} ):resize(1,2)
net.Wb[3] = torch.Tensor( {0.3,0.4} ):resize(1,2)

X = torch.Tensor({{0.2,0.7},{0.2,0.7}}):t()
T = torch.Tensor({{0,0},{1,1}}):resize(2,2)
feedForward(net, X)
--print(net.Y[net.nLayer])
DW, DWb = backpropagate(net, T)
print( DWb[2] )

print('something wrong')
