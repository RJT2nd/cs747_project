from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np

with open('log.txt', 'r') as log:
  data = log.read()

  temp = data.split('Training Loss[')[1:]
  train_losses = [ float(x.split(']')[0]) for x in temp]
  train_accs = [ float(x.split('] Acc[')[1].split(']')[0]) for x in temp]
  
  temp = data.split('Validation Loss[')[1:]
  val_losses = [ float(x.split(']')[0]) for x in temp]
  val_accs = [ float(x.split('] Acc[')[1].split(']')[0]) for x in temp]

  print(len(train_losses))
  print(len(train_accs))
  print(len(val_losses))
  print(len(val_accs))

  x = np.arange(1, 101)
  y_train = train_losses[:100]
  y_val = val_losses[:100]

  # smoothing the labels.
  X_Y_Spline = make_interp_spline(x, y_val)
  X_val = np.linspace(min(x), max(x), 500)
  Y_val = X_Y_Spline(X_val)
  
  X_Y_Spline = make_interp_spline(x, y_train)
  X_train = np.linspace(min(x), max(x), 500)
  Y_train = X_Y_Spline(X_train)

  # Plotting the Graph
  plt.plot(X_val, Y_val,label = "Validation Loss")
  plt.plot(X_train, Y_train,label = "Train Loss")
  plt.title("Cross Entropy Loss 100 epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
  
  
  x = np.arange(1, 101)
  y_train = train_accs[:100]
  y_val = val_accs[:100]

  # smoothing the labels.
  X_Y_Spline = make_interp_spline(x, y_val)
  X_val = np.linspace(min(x), max(x), 500)
  Y_val = X_Y_Spline(X_val)
  
  X_Y_Spline = make_interp_spline(x, y_train)
  X_train = np.linspace(min(x), max(x), 500)
  Y_train = X_Y_Spline(X_train)

  # Plotting the Graph
  plt.plot(X_val, Y_val,label = "Validation Accuracy")
  plt.plot(X_train, Y_train,label = "Train Accuracy")
  plt.title("Accuracy for 100 epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()