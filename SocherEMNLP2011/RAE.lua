NPROCESS = 14

--**************** RAE class ******************--
reAutoEncoder = {}
reAutoEncoder_mt = {__index = reAutoEncoder}

--****************** needed functions ******************--
-- generate a n x m matrix by uniform distibution within range [min,max]
function uniform( n, m, min, max )
	local M = torch.rand(n, m)
	M:mul(max-min):add(min)
	return M
end

-- normalization
-- input: 
-- 	X : n x m matrix, each colum is a n-dim vector
-- 	p : norm
-- output: norm_p of 
function normalize( X , p)
	if p == 1 then
		return torch.cdiv(X, torch.mm( torch.ones(X:size(1),1),  X:sum(1)  ))
	elseif p == 2 then	
		return torch.cdiv(X, torch.mm( torch.ones(X:size(1),1), torch.sqrt(torch.pow(X,p):sum(1))   ))
	else 
		return torch.cdiv(X, torch.mm( torch.ones(X:size(1),1), torch.pow(torch.pow(X,p):sum(1),1/p)  ))
	end
end

-- logistic function
function logistic( X )
	return torch.cdiv (
			torch.ones(X:size()), 
			torch.exp( torch.mul(X,-1) ):add(1)
		)
end

-- derivative of logistic function
-- input : 
-- 	logiX : logisitic(X)
function logisticPrime( logiX )
	local D = torch.mul(logiX, -1):add(1)
	return torch.cmul(D, logiX)
end

-- tanh function 
-- in range [-1,1]
function tanh( X )
	return torch.tanh(X)
end

-- derivative of norm2 tanh
-- input:
-- 	tanhX = tanh(X) must be a n x 1 vector 
-- output:
-- 	grad : n x n matrix
function norm2TanhPrime( tanhX )
	local dim = tanhX:size(1)
	local tanhX2 = torch.pow(tanhX,2)
	local OneMinusTanhX2 = torch.mul(tanhX2,-1):add(1)
	local nrm2 = math.sqrt(tanhX2:sum())

	local P1 = torch.diag( torch.mul(OneMinusTanhX2,1/nrm2):resize(dim)  )
	local P2 = torch.mm(tanhX, torch.cmul(tanhX, OneMinusTanhX2):t():mul(1/(nrm2*nrm2*nrm2)))
	return P1 - P2
end

--[[
function test()
	local eps = 1e-4
	local theta = 1e-8
	local X = torch.Tensor({1,2,3,4,5}):resize(5,1)
	local grad = norm2TanhPrime(tanh(X))

	for i = 1,5 do
		local index = {{i},1}
		X[index]:add(eps)
		local plus = normalize(tanh(X),2)
		X[index]:add(-2*eps)
		local minus = normalize(tanh(X),2)
		X[index]:add(eps)
		print( (plus-minus)/2/eps - grad[{{},i}]  )
	end
end

test()
]]

--************************* construction ********************--
-- create a new recursive autor encoder with a given structure
-- input:
-- 	struct = { dimension, nCategories, Lookup }
function reAutoEncoder:new( struct )
	local rae = {}
	local dim = struct.Lookup:size(1)
	local nCat = struct.nCategory

	rae.W11 = uniform(dim, dim, -1, 1)
	rae.W12 = uniform(dim, dim, -1, 1)
	rae.b1 = uniform(dim, 1, -1, 1)

	rae.W21 = uniform(dim, dim, -1, 1)
	rae.W22 = uniform(dim, dim, -1, 1)
	rae.b21 = uniform(dim, 1, -1, 1)
	rae.b22 = uniform(dim, 1, -1, 1)

	rae.WCat = uniform(nCat, dim, -1, 1)
	rae.bCat = uniform(nCat, 1, -1, 1)
	
	rae.L = struct.Lookup
	rae.func = struct.func
	rae.funcPrime = struct.funcPrime
	
	setmetatable(rae, reAutoEncoder_mt)
	return rae
end

--[[load network from file
function reAutoEncoder:load( filename , func, funcPrime )
	local file = torch.DiskFile.new(filename)
	local buff = file:readInt(3)
	local dim = buff[1]
	local dicLen = buff[2]
	local nCat = buff[3]

	local rae = {}
	setmetatable(rae, reAutoEncoder_mt)

	local Theta = torch.Tensor(file:readDouble(dim*dicLen + dim*dim*4 + dim*3 + dim*nCat + nCat))
	file:close()

	rae:unfold(Theta, dim, dicLen, nCat)
	rae.func = func or tanh
	rae.funcPrime = funcPrime or norm2TanhPrime
	rae.L = normalize(rae.L, 2)

	return rae
end
]]

-- save rae into a bin file
function reAutoEncoder:save( filename )
	local file = torch.DiskFile(filename, 'w')
	file:binary()
	file:writeObject(self)
	file:close()
end

-- create rae from file
function reAutoEncoder:load( filename )
	local file = torch.DiskFile(filename, 'r')
	file:binary()
	local rae = file:readObject()
	setmetatable(rae, reAutoEncoder_mt)
	file:close()
	return rae
end

-- fold parameters to a vector
function reAutoEncoder:fold( Model )
	local Params = {}
		Model = Model or {}
		Params[1] = Model.L or self.L
		Params[2] = Model.W11 or self.W11
		Params[3] = Model.W12 or self.W12
		Params[4] = Model.b1 or self.b1
		Params[5] = Model.W21 or self.W21
		Params[6] = Model.W22 or self.W22
		Params[7] = Model.b21 or self.b21
		Params[8] = Model.b22 or self.b22
		Params[9] = Model.WCat or self.WCat
		Params[10] = Model.bCat or self.bCat

	local dim = Params[1]:size(1)
	local dicLen = Params[1]:size(2)
	local nCat = Params[10]:size(1)

	local Theta = torch.zeros(dim*dicLen + dim*dim*4 + dim * 3 + dim*nCat + nCat)
	local i = 1
	for _,P in ipairs(Params) do
		local nElem = P:nElement()
		Theta[{{i,i+nElem-1}}] = P
		i = i + nElem
	end

	return Theta
end

-- unfold param-vector 
function reAutoEncoder:unfold( Theta , dim, dicLen, nCat )
	self.L = self.L or torch.Tensor(dim, dicLen)
	self.W11 = self.W11 or torch.Tensor(dim, dim)
	self.W12 = self.W12 or torch.Tensor(dim, dim)
	self.b1 = self.b1 or torch.Tensor(dim, 1)
	self.W21 = self.W21 or torch.Tensor(dim, dim)
	self.W22 = self.W22 or torch.Tensor(dim, dim)
	self.b21 = self.b21 or torch.Tensor(dim, 1)
	self.b22 = self.b22 or torch.Tensor(dim, 1)
	self.WCat = self.WCat or torch.Tensor(nCat, dim)
	self.bCat = self.bCat or torch.Tensor(nCat, 1)

	local Params = {}

		Params[1] = self.L
		Params[2] = self.W11
		Params[3] = self.W12
		Params[4] = self.b1
		Params[5] = self.W21
		Params[6] = self.W22
		Params[7] = self.b21
		Params[8] = self.b22
		Params[9] = self.WCat
		Params[10] = self.bCat

	local i = 1
	for _,P in ipairs(Params) do
		local nElem = P:nElement()
		P:copy(Theta[{{i,i+nElem-1}}])
		i = i + nElem
	end
end

--************************ forward **********************--
--input:
--	Sentence : a vector of word indices
--	Label : for classification 
--	config : {alpha, lambda}
--output:
--	tree
function reAutoEncoder:forward( Sentence, Label , config )

	local W11 = self.W11
	local W12 = self.W12
	local W21 = self.W21
	local W22 = self.W22
	local b1 = self.b1
	local b21 = self.b21
	local b22 = self.b22
	local WCat = self.WCat
	local bCat = self.bCat
	local L = self.L
	local func = self.func
	local funcPrime = self.funcPrime

	local dim = L:size(1)
	local dicLen = L:size(2)
	local nCat = self.bCat:size(1)

	local config = config or {alpha = 1, lambda = 1e04}	-- default is unsupervised learning
	local alpha = config.alpha
	local len = Sentence:size(1)

	Label = Label or torch.zeros(nCat, 1)	-- for testing with unknown label
	
	-- building tree
	curFeature = torch.zeros(dim,len)
	curSubtree = {}
	for i = 1,len do
		--print(Sentence[i]) print(L:size())
		feature = L[{{},{Sentence[i]}}]
		curFeature[{{},i}] = feature

		-- classify on single words
		local predict = normalize( torch.mm(WCat,feature):add( bCat ):exp(), 1 )
		local ecat = (-torch.cmul(Label, torch.log(predict))):sum() * (1-alpha) 
		curSubtree[i] = { feature = L[{{},{Sentence[i]}}], wordId = Sentence[i],
				predict = predict, ecat = ecat, cover = 1 , label = Label }
	end
	curCover = torch.ones(1,len)

	for ii = 1,len-1 do
		local curLen = curFeature:size(2)
		local C1 = curFeature[{{},{1,curLen-1}}]
		local C2 = curFeature[{{},{2,curLen}}]
		local c1Cover = curCover[{1,{1,curLen-1}}]  
		local c2Cover = curCover[{1,{2,curLen}}] 

		-- cal parent feature
		local unnormP = func( torch.mm(W11,C1):add(torch.mm(W12,C2)):add(torch.mm(b1,torch.ones(1,curLen-1)))  )
		local P = normalize(unnormP, 2)

		-- reconstruct child features
		local unRCC1 = func( torch.mm(W21, P):add(torch.mm(b21,torch.ones(1,curLen-1))) ) 
		local unRCC2 = func( torch.mm(W22, P):add(torch.mm(b22,torch.ones(1,curLen-1))) ) 
		local rcC1 = normalize(unRCC1, 2) 
		local rcC2 = normalize(unRCC2, 2) 
		local diff1 = torch.add(rcC1,-C1)
		local diff2 = torch.add(rcC2,-C2)

		-- cal reconstruction error
		local totalCover = torch.add(c1Cover, c2Cover)
		local Erec = torch.add(
			torch.cmul(torch.cdiv(c1Cover, totalCover), torch.pow(diff1,2):sum(1)),
			torch.cmul(torch.cdiv(c2Cover, totalCover), torch.pow(diff2,2):sum(1)) ):mul(alpha)

		-- extract the min error
		local _,pos = torch.min(Erec,1) 
		pos = pos[1] --print(pos)
		local curIndex = {{},{pos}}
		local p = P[curIndex] 
		local unp = unnormP[curIndex] 
		local erec = Erec[{pos}] 
		local c1cover = c1Cover[{pos}] 
		local c2cover = c2Cover[{pos}]
		local cover = c1cover + c2cover

		-- compute classification error
		local predict = normalize( torch.mm(WCat,p):add( bCat ):exp(), 1 ) 
		local ecat = (-torch.cmul(Label, torch.log(predict))):sum() * (1-alpha)

		-- update tree
		local tree = {
			feature = p:clone(),
			unnormFeature = unp:clone(),

			unRCC1 = unRCC1[curIndex]:clone(),
			unRCC2 = unRCC2[curIndex]:clone(),
			rcC1Feature = rcC1[curIndex]:clone(),
			rcC2Feature = rcC2[curIndex]:clone(),
			c1Diff = diff1[curIndex]:clone():mul(2*alpha*c1cover/cover),
			c2Diff = diff2[curIndex]:clone():mul(2*alpha*c2cover/cover),

			predict = predict:clone(),
			label = Label,
			
			erec = erec,
			ecat = ecat,
			cover = cover,

			child1 = curSubtree[pos],
			child2 = curSubtree[pos+1]
		}

		-- for the next loop
		newFeature = torch.zeros(dim, curLen-1)
		if pos > 1 then newFeature[{{},{1,pos-1}}] = curFeature[{{},{1,pos-1}}] end
		newFeature[{{},pos}] = p
		if pos+2 <= curLen then newFeature[{{},{pos+1,curLen-1}}] = curFeature[{{},{pos+2,curLen}}] end 
		curFeature = newFeature

		newCover = torch.zeros(1,curLen-1)
		if pos > 1 then newCover[{1,{1,pos-1}}] = curCover[{1,{1,pos-1}}] end
		newCover[{1,pos}] = cover
		if pos+2 <= curLen then newCover[{1,{pos+1,curLen-1}}] = curCover[{1,{pos+2,curLen}}] end
		curCover = newCover 

		table.remove(curSubtree, pos+1)
		curSubtree[pos] = tree
	end
	return curSubtree[1]
end

--*********************** backpropagate *********************--
-- only for one tree/sentence
-- input:
-- 	tree : result of the parse function
-- 	config :
-- output:
function reAutoEncoder:backpropagate( tree, config , grad )

	local dim = self.L:size(1)
	local dicLen = self.L:size(2)
	local nCat = self.bCat:size(1)

	local GW2 = torch.Tensor(2*dim + nCat,dim)
	GW2[{{1,dim},{}}] = self.W21
	GW2[{{dim+1,2*dim},{}}] = self.W22
	GW2[{{2*dim+1,2*dim+nCat},{}}] = self.WCat

	local gradL = grad.L
	local gradW11 = grad.W11
	local gradW12 = grad.W12
	local gradb1 = grad.b1
	local gradW21 = grad.W21
	local gradW22 = grad.W22
	local gradb21 = grad.b21
	local gradb22 = grad.b22
	local gradWCat = grad.WCat
	local gradbCat = grad.bCat
	local cost = 0

	local propagate
	propagate = function( tree , diff , W , gradZp) 
	-- diff = reconstrVector - trueVector , W = W11 if left-child else W12
	-- gradZp : gradZ at the parent node
		local W11 = self.W11
		local W12 = self.W12
		local W21 = self.W21
		local W22 = self.W22
		local b1 = self.b1
		local b21 = self.b21
		local b22 = self.b22
		local WCat = self.WCat
		local bCat = self.bCat
		local W2 = GW2

		local L = self.L
		local func = self.func
		local funcPrime = self.funcPrime

		local dim = L:size(1)
		local dicLen = L:size(2)
		local nCat = self.bCat:size(1)

		local alpha = config.alpha
		local mDiff = -diff

		-- for internal node
		if tree.child1 ~= nil and tree.child2 ~= nil then
			-- compute cost
			cost = cost + tree.erec + tree.ecat
			
			-- compute gradZ
			local gradZ21 = torch.mm(tree.c1Diff:t(), funcPrime(tree.unRCC1)):t()
			local gradZ22 = torch.mm(tree.c2Diff:t(), funcPrime(tree.unRCC2)):t()
			local gradZCat = torch.add(tree.predict,-tree.label):mul(1-alpha )
 
			local gradZ2 = torch.Tensor(2*dim+nCat, 1)
			gradZ2[{{1,dim},{}}] = gradZ21
			gradZ2[{{dim+1,2*dim},{}}] = gradZ22
			gradZ2[{{2*dim+1,2*dim+nCat},{}}] = gradZCat
			local gradZ = torch.mm(funcPrime(tree.unnormFeature):t(), torch.mm(W2:t(), gradZ2 ):add( torch.mm(W:t(), gradZp) ):add(mDiff) )

			-- compute gradient
			gradW21:add(torch.mm(gradZ21, tree.feature:t()))
			gradb21:add(gradZ21)
			gradW22:add(torch.mm(gradZ22, tree.feature:t()))
			gradb22:add(gradZ22)
			gradWCat:add(torch.mm(gradZCat, tree.feature:t()))
			gradbCat:add(gradZCat)

			gradW11:add(torch.mm(gradZ, tree.child1.feature:t())) 
			gradW12:add(torch.mm(gradZ, tree.child2.feature:t()))
			gradb1:add(gradZ)

			-- propagate to its children
			propagate( tree.child1, tree.c1Diff, W11, gradZ  )
			propagate( tree.child2, tree.c2Diff, W12, gradZ )

		else -- leaf
			-- compute cost
			cost = cost + tree.ecat

			-- compute gradZ
			local gradZCat = torch.add(tree.predict,-tree.label):mul( 1-alpha )
			local gradZ = torch.mm(W:t(), gradZp):add(mDiff):add( torch.mm( WCat:t(), gradZCat  )  )

			-- compute gradient
			gradWCat:add(torch.mm(gradZCat, tree.feature:t()))
			gradbCat:add(gradZCat)
			gradL[{{},{tree.wordId}}]:add( gradZ )
		end
	end
	
	propagate(tree, torch.zeros(dim,1), torch.zeros(dim,dim), torch.zeros(dim,1))

	return cost
end

--************************ compute cost and gradient *****************--
--input:
--output:
require 'parallel'

-- worker function 
function worker()

	require 'RAE'
	local data = parallel.parent:receive()

	local Sentences = data.Sentences
	local Labels = data.Labels
	local rae = data.rae
	local nSen = #Sentences
	local config = data.config

	local grad = {
		L = torch.zeros(rae.L:size()),
		W11 = torch.zeros(rae.W11:size()),
		W12 = torch.zeros(rae.W12:size()),
		b1 = torch.zeros(rae.b1:size()),
		W21 = torch.zeros(rae.W21:size()),
		W22 = torch.zeros(rae.W22:size()),
		b21 = torch.zeros(rae.b21:size()),
		b22 = torch.zeros(rae.b22:size()),
		WCat = torch.zeros(rae.WCat:size()),
		bCat = torch.zeros(rae.bCat:size())
	}

	local cost = 0
	local Trees = {}
	for i = 1, nSen do
		local Sen = Sentences[i]
		local Label = Labels[i]
		local tree = reAutoEncoder.forward(rae, Sen, Label, config)
		cost = cost + reAutoEncoder.backpropagate(rae, tree, config , grad)
		Trees[i] = tree

		--if math.mod(i,100) == 0 then
		--	print('i = ' .. i .. ' : cost = ' .. cost )
		--end
	end

	parallel.parent:send( { cost = cost, grad = reAutoEncoder.fold(rae, grad) , Trees = Trees } )
end
	
-- parent call
function parent(param)

	local Sentences = param.Sentences
	local Labels = param.Labels
	local nSen = #Sentences
	local rae = param.rae

	-- split data
	local size = math.ceil(nSen / NPROCESS)
	local children = parallel.sfork(NPROCESS)
	children:exec(worker)

	-- send data
	for i = 1, NPROCESS do
		local data = {Sentences = {}, Labels = {}, rae = param.rae, config = param.config}
		for j = 1,size do
			local id = (i-1)*size+j
			if id > nSen then break end
			data.Sentences[j] = Sentences[id]
			data.Labels[j] = Labels[id]
		end
		children[i]:send(data)
	end

	-- receive results
	for i = 1, NPROCESS do
		local reply = children[i]:receive()
		param.totalCost = param.totalCost + reply.cost
		if param.totalGrad == nil then
			param.totalGrad = reply.grad
		else
			param.totalGrad:add(reply.grad)
		end

		if param.Trees == nil then param.Trees = {} end
		for j = 1,#reply.Trees do
			param.Trees[#param.Trees+1] = reply.Trees[j]
		end
	end

	children:sync()

	-- finalize
	local M = param.rae:fold()
	param.totalCost = param.totalCost * (1/nSen) + param.config.lambda/2 * torch.pow(M,2):sum()
	param.totalGrad:mul(1/nSen):add(torch.mul(M,param.config.lambda))
end

function reAutoEncoder:computeCostAndGrad( Sentences, Labels , config )
	
	local param = {
		rae = self,
		config = config,
		Sentences = Sentences,
		Labels = Labels,
		totalCost = 0,
		totalGrad = nil
	}


	local ok,err = pcall(parent, param)
	if not ok then 	print(err) end
	--parallel.close()
	
	return param.totalCost, param.totalGrad, param.Trees
end

-- check gradient
function reAutoEncoder:checkGradient(Sentences, Labels, config)
	local epsilon = 1e-4
	local theta = 1e-8

	local dim = self.L:size(1)
	local dicLen = self.L:size(2)
	local nCat = self.bCat:size(1)

	local good = true
	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(Sentences, Labels, config)

	--for i = dim*dicLen-2,dim*dicLen+10,2 do
	for i = 1,dim*dicLen + 4*dim*dim + 3*dim + dim*nCat + nCat do
		local index = {{i}}
		Theta[index]:add(epsilon)
		self:unfold(Theta)
		local costPlus,_ = self:computeCostAndGrad(Sentences, Labels, config)
		
		Theta[index]:add(-2*epsilon)
		self:unfold(Theta)
		local costMinus,_ = self:computeCostAndGrad(Sentences, Labels, config)
		Theta[index]:add(epsilon)
		self:unfold(Theta)

		local diff = math.abs( (costPlus - costMinus) / (2*epsilon) - gradTheta[i] )
		print(diff)

		if diff > theta then 
			good = false
			--break
		end
	end

	return good
end

--************************** parse *************************
function reAutoEncoder:parse( Sentences )
	local nSen = #Sentences
	local nCat = self.bCat:size(1)

	local Labels = {}
	for i = 1, nSen do
		Labels[i] = torch.zeros(nCat, 1)
	end
	config = {alpha = 1, lambda = 0}
	
	local _,_,Trees = self:computeCostAndGrad( Sentences, Labels, config )
	return Trees
end

function reAutoEncoder:getFeature( tree ) 
	local dim = self.L:size(1)
	local innerFeat = torch.zeros(dim,1)
	
	local acc 
	acc = function( t  ) 
		innerFeat:add(t.feature)
		if t.child1 ~= nil and t.child2 ~= nil then
			acc( t.child1 )
			acc( t.child2 )
		end
	end

	acc(tree)
	innerFeat:mul(1 / (2*tree.cover-1))

	local feature = torch.Tensor(2*dim,1)
	feature[{{1,dim},{1}}] = tree.feature
	feature[{{dim+1,2*dim},{1}}] = innerFeat

	return feature
end

--*********************** train the final classifier *****************
require 'mlp'
function reAutoEncoder:trainFinalClassifier( Sentences, Labels, optFunc, optFuncConfig )
	local dim = self.L:size(1)
	local nCat = self.bCat:size(1)

	local struct =  { 
		{size = 2*dim, f = nil, bias = 0},
		{size = nCat, f = AtvFunc.normExp, bias = 1}
	}
	local classifier = mlp:new( struct , CostFunc.crossEntropyCost )
	classifier:initWeights()
	
	local Trees = self:parse(Sentences)
	local nSample = #Trees
	local X = torch.Tensor(2*dim, nSample)
	local T = torch.Tensor(nCat, nSample)

	for i = 1 , nSample do
		X[{{},{i}}] = self:getFeature(Trees[i])
		T[{{},{i}}] = Labels[i]
	end

	classifier:train(X, T, nSample, optFunc, optFuncConfig)
	return classifier
end

--******************************** test **************************
function reAutoEncoder:test( classifier, Sentences, Labels)

	local dim = self.L:size(1)
	local nCat = self.bCat:size(1)

	local Trees = self:parse(Sentences)
	local nSample = #Trees
	local X = torch.Tensor(2*dim, nSample)
	local T = torch.Tensor(nCat, nSample)

	for i = 1 , nSample do
		X[{{},{i}}] = self:getFeature(Trees[i])
		T[{{},{i}}] = Labels[i]
	end
	
	local Y = classifier:feedforward(X)
 	
	-- error rate
	local errorRate = function( Y, T )
        	local temp, IY = torch.max(Y,1)
	        temp, IT = torch.max(T,1)
        	return torch.ne(IY,IT):sum() / Y:size()[2]
	end


	return 1 - errorRate(Y,T)
end

--******************************* train networks *************************
---- optFunc from 'optim' package
function reAutoEncoder:train( Data, batchSize, optFunc, optFuncState, config)
	local trainData = Data.train
	local testData = Data.test

	local nSample = #trainData.Sentences
	local j = 0

	--print(self:test(testData.Sentences, testData.Labels))

	--return [cost,gradient]
	local iter = 1
	timer = torch.Timer()
	
	local function func( M )
		self:unfold(M)

		-- extract data
		j = j + 1
		--if j > nSample/batchSize then
		--	j = 1
		--end
		--local subSentences = Sentences
		--local subLabels = Labels
		local cost, Grad = self:computeCostAndGrad(trainData.Sentences, trainData.Labels, config)

		-- for visualization
		if math.mod(iter,1) == 0 then
			print('--- iter: ' .. iter)
			print('cost: ' .. cost)
			print('time: ' .. timer:time().real) timer = torch.Timer()
			--print(self:checkGradient(subX,subT))
		end
		if math.mod(iter,10) == 0 then
			print('accuracy = ' .. self:test(testData.Sentences, testData.Labels))
			self:save('model.' .. math.floor(iter / 10))
		end
		io.flush()

		iter = iter + 1
		collectgarbage()
		return cost, Grad
	end

	for i = 1,1 do
		local M = optFunc(func, self:fold(), optFuncState, optFuncState)
		self:unfold(M)
	end
end

--*********************************** main ******************************--
function test ()
	local rae = reAutoEncoder:load( 'rae.txt' )
	local Sentences = {
			torch.Tensor({2,3}),
			torch.Tensor({1,3,4}),
			torch.Tensor({1,2,3,5,6}),
			torch.Tensor({2})
		}
	local Labels = {
			torch.Tensor({1,0}):resize(2,1),
			torch.Tensor({0,1}):resize(2,1),
			torch.Tensor({1,0}):resize(2,1),
			torch.Tensor({1,0}):resize(2,1)
		}


	config = {lambda = 1e-4, alpha = 0.2}
	--print( rae:checkGradient( {Sentences[1],Sentences[2]}, {Labels[1],Labels[1]} , config)  ) 
	--print( rae:checkGradient( {Sentences[3]}, Labels , config)  ) 
	--print( rae:checkGradient( {Sentences[4]}, Labels , config)  ) 
	print(rae:checkGradient(Sentences, Labels, config))
end

--test()
