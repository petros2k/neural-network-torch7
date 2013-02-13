require 'wordMeanRep'

Util = {}

function Util.split(str, sep)
	local sep, fields = sep or ":", {}
	local pattern = string.format("([^%s]+)", sep)
	str:gsub(pattern, function(c) fields[#fields+1] = c end)
	return fields
end

function Util.unrequire(m)
	package.loaded[m] = nil
	_G[m] = nil
end

return Util
