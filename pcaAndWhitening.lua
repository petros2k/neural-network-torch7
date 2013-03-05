--********************** pca *********************--
-- assume that X is already normalized (0 mean and 1 var for each feature)
pca = {}

-- extract eigenvectors and eigenvalues
function pca.pca( X )
	local Sigma = torch.mm(X, X:t()):mul(1 / X:size(2))
	local U,S,V = torch.svd(Sigma)
	return U,S
end

-- get number of retained dim such that the sum retained var is not less than pVarRetain
function pca.getNRetainDim( eigVec, eigValue, pVarRetain)
	local sumVar = eigValue:sum()
	local ret
	local newSumVar = 0
	for i = 1,eigValue:size(1) do
		newSumVar = newSumVar + eigValue[{i,1}]
		--require 'debugger'
		--if _DEBUG_ then print(newSumVar) pause() end
		if newSumVar / sumVar > pVarRetain then 
			ret = i
			break 
		end
	end

	return ret
end

-- reduce dim
function pca.approximate( X, eigVec, nRetainDim)
	local newEVec = eigVec[{{},{1,nRetainDim}}]:clone()
	return torch.mm(newEVec:t(), X)
end

-- recover
function pca.recover( appX, eigVec )
	local nDim = appX:size(1)
	return torch.mm(eigVec[{{},{1,nDim}}], appX)
end 


--***************************************** test *****************************************--
iRow = 512
iCol = 512
pRow = 20
pCol = 20
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
	local Mean = torch.mm(torch.Tensor(X:size()[1],1):fill(1), X:mean(1))
	Mean:mul(-1)
	X:add(Mean)

	return X
end

require 'image'

function test()
	print('load data...')
	local X = loadData('image_raw')
	local eigVec, eigVal = pca.pca(X)
	eigVal:resize(eigVal:size(1),1)
	
	local nSample = X:size(2)

	-- check covar
	local Xrot = torch.mm(eigVec:t(),X)
	local Sig = torch.mm(Xrot, Xrot:t()):mul(1 / nSample)
	image.display(Sig)

	-- compress
	local nRetainDim99 = pca.getNRetainDim(eigVec, eigVal, 0.99)
	local Xpca99 = pca.recover(pca.approximate( X, eigVec, nRetainDim99 ),eigVec)
	local nRetainDim90 = pca.getNRetainDim(eigVec, eigVal, 0.90)
	local Xpca90 = pca.recover(pca.approximate( X, eigVec, nRetainDim90 ),eigVec)

	-- whitening
	local epsilon = 0.1
	local M = torch.mm(eigVal, torch.Tensor(1,nSample):fill(1)):add(epsilon):sqrt()
	local Xpcawhite = torch.cdiv(Xrot,M)
	Sig = torch.mm(Xpcawhite, Xpcawhite:t()):mul(1 / nSample)
	image.display(Sig)

	-- ZCA whitening
	local Xzcawhite = torch.mm(eigVec, Xpcawhite)

	-- visualization	
	local IX = torch.Tensor(10,pRow,pCol):fill(0)
	local IX99 = torch.Tensor(10,pRow,pCol):fill(0)
	local IX90 = torch.Tensor(10,pRow,pCol):fill(0)
	local IXzca = torch.Tensor(10,pRow,pCol):fill(0)
	for i = 1,10 do
		IX[{i,{},{}}] = X[{{},i}]:clone():resize(pRow,pCol)
		IX99[{i,{},{}}] = Xpca99[{{},i}]:clone():resize(pRow,pCol)
		IX90[{i,{},{}}] = Xpca90[{{},i}]:clone():resize(pRow,pCol)
		IXzca[{i,{},{}}] = Xzcawhite[{{},i}]:clone():resize(pRow,pCol)
	end
	image.display({image=IX,zoom=2})
	image.display({image=IX99,zoom=2})
	image.display({image=IX90,zoom=2})
	image.display({image=IXzca,zoom=2})
end

test()
