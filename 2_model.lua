
------------------------------
-- library
------------------------------

require 'torch'
require 'nn'
require 'cunn'

------------------------------
-- function
------------------------------

function newmodel()

	-- 10-class problem
	local noutputs = 10

	-- input dimensions
	local nfeats = 1
	local width = 28
	local height = 28
	local ninputs = nfeats*width*height

	-- hidden units:
	local nstates = {64,64,256,1024,256}

	-- model:
	local model = nn.Sequential()

	-- stage 1 : Convolution // 1x28x28 -> 64x24x24 -> 64x12x12
	model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], 5, 5))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Dropout(0.1))

	-- stage 2 : Convolution // 64x12x12 -> 64x12x12 -> 64x6x6
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], 5, 5, 1, 1, 2))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Dropout(0.2))

	-- stage 3 : Convolution // 64x6x6 -> 256x6x6 -> 256x3x3
	model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], 3, 3, 1, 1, 1))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Dropout(0.3))

	-- stage 4 : Convolution // 256x3x3 -> 1024x1x1
	model:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], 3, 3))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.4))

	-- stage 5 : FC // 1024 -> 256
	model:add(nn.View(nstates[4]))
	model:add(nn.Linear(nstates[4], nstates[5]))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	-- stage 6 : FC // 256 -> 10
	model:add(nn.Linear(nstates[5], noutputs))
	model:add(nn.LogSoftMax())

	print(model)

	return model

end

------------------------------
-- main
------------------------------

model = newmodel()

-- loss function
criterion = nn.ClassNLLCriterion()
print '==> here is the loss function:'
print(criterion)
