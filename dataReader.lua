DataReader = {}

--*************************** load data ******************* --
-- read data from file
--[[
- line 1: [num_of_examples] [input_dim] [output_dim]
- line 2: 
]]--
function DataReader.loadCompleteData( path )
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
	Data.nSample = buff[1]
	Data.nInDim = buff[2]
	Data.nOutDim = buff[3]
	Data.X = {}
	Data.T = {}

-- read next
	for i = 1, Data.nSample do
		Data.X[i] = {}
		local xlen = file:readInt(1)[1]
		for j = 1,xlen do
			local idx = file:readInt(1)[1]
			local value = file:readDouble(1)[1]
			Data.X[i][idx] = value
		end

		Data.T[i] = {}
		local tlen = file:readInt(1)[1]
		for j = 1,tlen do
			local idx = file:readInt(1)[1]
			local value = file:readDouble(1)[1]
			Data.T[i][idx] = value
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
	local nInDim = 500
	local nOutDim = 2
	file:writeInt(nSample)
	file:writeInt(nInDim)
	file:writeInt(nOutDim)

	for i = 1,nSample do
		local X = torch.rand(nInDim)
		for j = 1,nInDim do
			file:writeDouble(X[j])
		end

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

return DataReader
