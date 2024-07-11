import torch
import sys

model = torch.load(sys.argv[1])

total_params = 0

for key in model.keys():
	if sys.argv[2] == '1':
		print(model[key].size())
	for index, dim in enumerate(model[key].size()):
		if index == 0:
			new_params = dim
		else:
			new_params *= dim
	total_params += new_params
print(f'This model has {total_params} total parameters')
