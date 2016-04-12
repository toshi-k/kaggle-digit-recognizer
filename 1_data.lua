
------------------------------
-- library
------------------------------

require 'torch'
require 'csvigo'
require 'nn'
require 'image'

------------------------------
-- function
------------------------------

function read_data(path, istrain)
	local csv_data = csvigo.load{path = path}

	local data_nrow = #csv_data["pixel0"]
	print("data_nrow:" .. data_nrow)

	local label
	if istrain then
		label = torch.Tensor(csv_data["label"])
	end
	local data = torch.Tensor(data_nrow, 28, 28)

	for pixel, ver in pairs(csv_data) do
		if pixel ~= "label" then

			npix = string.sub(pixel, 6) + 0
			i = math.floor(npix / 28) + 1;
			j = npix % 28 + 1;

			data[{ {},i,j }] = torch.Tensor(ver)
		end
	end

	data = data:float()
	if istrain then
		label = label:int()
	end

	return data, label
end

------------------------------
-- main
------------------------------

-- read data: -----

train_data, train_label = read_data("dataset/train.csv", true)
test_data, _ = 	read_data("dataset/test.csv", false)

-- normalzie globally: -----

mean_ = train_data:mean()
std_ = train_data:std()

train_data:add(-mean_)
train_data:div(std_)

test_data:add(-mean_)
test_data:div(std_)

-- normalzie locally: -----

neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

for i = 1,train_data:size(1) do
	train_data[{{i}}] = normalization:forward(train_data[{{i}}])
end

for i = 1,test_data:size(1) do
	test_data[{{i}}] = normalization:forward(test_data[{{i}}])
end
