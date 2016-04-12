
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'cunn'

------------------------------
-- function
------------------------------

function save_submission(ImageID, Label)

	local submission = {}
	submission["ImageId"] = ImageID
	submission["Label"] = Label

	local fp = io.open("submission_train" .. string.format("%.3f", train_score) .. ".csv", "w")

	local headers = {"ImageID", "Label"}
	local headwrite = table.concat(headers, ",")
	fp:write(headwrite.."\n")

	for i=1,#ImageID do
		local row = {ImageID[i], Label[i]}
		local rowwrite = table.concat(row, ",")
		fp:write(rowwrite.."\n")
	end

	fp:close()
end

function test()

	local test_nrow = test_data:size(1)

	-- local vars
	local ImageID = {}
	local Label = {}

	-- set model to evaluate mode
	model:evaluate()

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	for t = 1,test_nrow do
		-- disp progress
		xlua.progress(t, test_nrow)

		-- get new sample
		local input = test_data[{{t},}]
		input = input:cuda()

		-- test sample
		local pred = model:forward(input)
		local _,p = torch.max(pred,1)

		table.insert(ImageID, tostring(t))
		table.insert(Label, tostring(p[1]-1))
	end

	save_submission(ImageID, Label)
end
