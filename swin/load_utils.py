import torch
from collections import OrderedDict

def single(save_dir):
	try:
		state_dict = torch.load(save_dir)['state_dict']
		new_state_dict = OrderedDict()

		for k, v in state_dict.items():
			name = k[7:]
			new_state_dict[name] = v
	except KeyError:
		new_state_dict = torch.load(save_dir)

	return new_state_dict