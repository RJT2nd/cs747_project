import torch
import numpy as np
import torchaudio
import os

def load_csv_and_convert(csv_filename='Data/features_30_sec.csv', mel=False):
  data = np.loadtxt(csv_filename, delimiter=',', dtype=str)
  out_dir = 'Data/hd_specs'
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  tensor_paths = convert_to_spec(data[1:], out_dir, mel=mel)
  X = tensor_paths
  Y = data[1:, -1]
  return X, Y

# def load_prerendered_csv(csv_filename='Data/features_30_sec.csv'):
#   data = np.loadtxt(csv_filename, delimiter=',', dtype=str)
#   tensor_paths = convert_to_spec(data[1:], out_dir)
#   X = tensor_paths
#   Y = data[1:, -1]
#   return X, Y

def convert_to_spec(data, out_dir, in_dir='Data/genres_original', mel=False):
  in_paths = []
  for row in data:
    in_path = in_dir + '/' + row[0].split('.')[0] + '/' + row[0]
    in_paths.append(in_path)
  out_paths = generate_spectrograms(in_paths, out_dir, mel)
  return out_paths

def generate_spectrograms(in_paths, out_dir, mel=False):
  out_paths = []
  for i, in_path in enumerate(in_paths):
    out_path = out_dir + '/' + in_path.split('/')[-1] + '.pt'
    if not os.path.exists(out_path):
      generate_stft(in_path, out_path, mel)
    out_paths.append(out_path)
  return out_paths

def generate_stft(in_path, out_path, mel=False):
  print(in_path)
  wf, sr = torchaudio.load(in_path)
  if(mel):
    transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=4096)
    spec = transform(wf)
  else:
    spec = torch.squeeze(torch.stft(wf, n_fft=2048).permute(0, 3, 1, 2))
  print(spec.size())
  torch.save(spec, out_path)