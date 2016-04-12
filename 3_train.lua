
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

------------------------------
-- main
------------------------------

model:cuda()
criterion:cuda()

-- classes
train_label:add(1)
classes = {0,1,2,3,4,5,6,7,8,9}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
if model then
	parameters,gradParameters = model:getParameters()
end

print '==> configuring optimizer'
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.adam

print '==> defining training procedure'
function train()

	local train_nrow = train_data:size(1)

	-- epoch tracker
	epoch = epoch or 1

	-- set model to training mode
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(train_nrow)

	-- do one epoch
	print(sys.COLORS.cyan .. '==> training on train set: # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,train_nrow,opt.batchSize do

		-- disp progress
		xlua.progress(t, train_nrow)

		-- create mini batch
		local inputs = torch.Tensor(math.min(t+opt.batchSize-1,train_nrow) - t + 1, 1, 28, 28)
		local targets = torch.Tensor(math.min(t+opt.batchSize-1,train_nrow) - t + 1)

		-- create mini batch
		local count = 1
		for i = t,math.min(t+opt.batchSize-1,train_nrow) do
			-- load new sample
			local input = train_data[{{shuffle[i]}}]
			local target = train_label[shuffle[i]]

			inputs[{count}] = input
			targets[{count}] = target

			count = count + 1
		end

		inputs = inputs:cuda()
		targets = targets:cuda()

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- estimate f
			local output = model:forward(inputs)
			local f = criterion:forward(output, targets)

			-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward(inputs, df_do)

			-- update confusion
			for i = 1,inputs:size(1) do
				confusion:add(output[{i}], targets[i])
			end

			-- normalize gradients and f(X)
			gradParameters:div(inputs:size(1))

			-- f is the average of all criterions
			f = f/inputs:size(1)

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)
	end
	xlua.progress(train_nrow, train_nrow)

	-- print confusion matrix
	print(confusion)

	-- get train score
	train_score = confusion.totalValid

	-- save/log current net
	local filename = "model.net"
	os.execute("mkdir -p ./" .. sys.dirname(filename))
	print("==> saving model to" ..filename)
	torch.save(filename, model:clearState())

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end
