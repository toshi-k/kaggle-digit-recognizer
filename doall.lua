
------------------------------
-- library
------------------------------

require 'nn'
require 'cunn'

------------------------------
-- settings
------------------------------

cmd = torch.CmdLine()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- training:
cmd:option('-learningRate', 1e-4, 'learning rate at t=0')
cmd:option('-batchSize', 16, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:text()
opt = cmd:parse(arg or {})

------------------------------
-- main
------------------------------

torch.setdefaulttensortype('torch.FloatTensor')

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- tic:
start_time = os.date()

-----

dofile "1_data.lua"
dofile "2_model.lua"
dofile "3_train.lua"
dofile "4_test.lua"

-----

print '==> training!'
for i = 1,40 do
	train()
end

test()

-- tac:
print '===> computation time'
end_time = os.date()
print("start_time:\t" .. start_time)
print("end_time:\t" .. end_time)
