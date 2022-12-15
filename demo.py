from model.RobNet import RobNetMel
import matplotlib.pyplot as plt
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

device = torch.device('cuda:0')

model = RobNetMel().to(device)
model.load_state_dict(torch.load('checkpoints/checkpoint.pt')["model_state_dict"])
model.eval()

print(model)

return_nodes = {
    "block4": "block4"
}
model2 = create_feature_extractor(model, return_nodes=return_nodes)

blues = torch.load('Data/hd_specs/blues.00000.wav.pt')
classical = torch.load('Data/hd_specs/classical.00000.wav.pt')
country = torch.load('Data/hd_specs/country.00000.wav.pt')
disco = torch.load('Data/hd_specs/disco.00000.wav.pt')
hiphop = torch.load('Data/hd_specs/hiphop.00000.wav.pt')
jazz = torch.load('Data/hd_specs/jazz.00000.wav.pt')
metal = torch.load('Data/hd_specs/metal.00000.wav.pt')
pop = torch.load('Data/hd_specs/pop.00000.wav.pt')
reggae = torch.load('Data/hd_specs/reggae.00000.wav.pt')
rock = torch.load('Data/hd_specs/rock.00000.wav.pt')

class_dict = {
  'blues': 0,
  'classical': 1,
  'country': 2,
  'disco': 3,
  'hiphop': 4,
  'jazz': 5,
  'metal': 6,
  'pop': 7,
  'reggae': 8,
  'rock': 9,
}


for i, tensor in enumerate([blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock]):
  tensor = tensor.to(device)
  # probs = torch.argmax(model(tensor[None, :128, :320]))
  # print(f'Expected {i}, predicted {torch.argmax(probs, dim=0).item()}')
  input = tensor.detach()[0,:,:].cpu()
  features = model2(tensor[None, :128, :320])
  print(features["block4"].size())
  print(input.shape)
  print('\n\nTensor', i)
  plt.imshow(input, origin='lower')
  plt.show()
  for j in range(features["block4"].size()[1]):
    plt.imshow(features["block4"].detach().cpu().numpy()[0, j,:,:], origin="lower")
    plt.show()
  # print(f'Expected {i}, predicted {torch.argmax(probs, dim=0).item()}')
