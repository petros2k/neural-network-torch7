DataReader = {}

--*************************** load data ******************* --
-- read data from file
--[[
- line 1: [num_of_examples] [input_dim] [output_dim]
- line 2: 
]]--
function DataReader.loadNormalData( path )
	local file = torch.DiskFile.new(path)
	local Data = {}
	
-- read the first 3 integer numbers
	local buff = file:readInt(3)
	local nSample = buff[1]
	local nInDim = buff[2]
	local nOutDim = buff[3]

	Data.X = torch.Tensor(nInDim, nSample)
	Data.T = torch.Tensor(nOutDim, nSample)

-- read next
	for i = 1, nSample do
		local X = torch.Tensor(file:readDouble(nInDim)):resize(nInDim, 1)
		Data.X[{{},i}] = X
		local T = torch.Tensor(file:readDouble(nOutDim)):resize(nOutDim, 1)
		Data.T[{{},i}] = T
	end
	
-- finish
	file:close()
	return Data
end

-- load compact data
function DataReader.loadCompactData( path )
	local file = torch.DiskFile.new(path)
	local Data = {}
	
-- read the first 3 integer numbers
	local buff = file:readInt(3)
	local nSample = buff[1]
	local nInDim = buff[2]
	local nOutDim = buff[3]

	print(nSample)
	print(nInDim)
	Data.X = torch.Tensor(nInDim, nSample):fill(0)
	Data.T = torch.Tensor(nOutDim, nSample):fill(0)

-- read next
	for i = 1, nSample do
		local xlen = file:readInt(1)[1]
		for j = 1,xlen do
			local idx = file:readInt(1)[1]
			local value = file:readDouble(1)[1]
			Data.X[idx][i] = value
		end

		local tlen = file:readInt(1)[1]
		for j = 1,tlen do
			local idx = file:readInt(1)[1]
			local value = file:readDouble(1)[1]
			Data.T[idx][i] = value
		end
	end
	
-- finish
	file:close()
	return Data
end


-- create fake data to test
function DataReader.createData( path )
	local file = torch.DiskFile.new(path, "w")
	local nSample = 1000
	local nInDim = 2
	local nOutDim = 2
	file:writeInt(nSample)
	file:writeInt(nInDim)
	file:writeInt(nOutDim)

	for i = 1,nSample do
		local X = torch.rand(2)
		file:writeDouble(X[1])
		file:writeDouble(X[2])

		local Y = torch.Tensor({0,1})
		if X[1] * X[2] < 0.1 then
		   Y = torch.Tensor({1,0})
		end
		
		file:writeDouble(Y[1])
		file:writeDouble(Y[2])
	end

	file:close()
end

--*****************************************************************

data = DataReader.loadCompactData("data/data.test")

return DataReader