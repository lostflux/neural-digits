

import numpy as np
import matplotlib.pyplot as plt


def visualize(x, y_true, /, y_pred=None, rows=2, cols=5, split="train"):
  
  images = np.reshape(x, (-1, 28, 28))  
  
  # get values from one-hot encoded vectors
  if y_true.ndim == 2:
    y_true = np.argmax(y_true, axis=1)

  fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols,2*rows))
  
  for pos in range(rows * cols):
    
    ax = axes[pos//cols, pos%cols]
    
    ax.imshow(images[pos], cmap="gray")
    
    if split == "train":
      ax.set_title(f"Label: {y_true[pos]}")
    elif split == "test":
      col = "green"
      if y_predicted[pos] != y_true[pos]:
        col = "red"
      ax.set_title(f"Label: {y_true[pos]} vs {y_predicted[pos]}", color = col)
          
  plt.tight_layout()
  plt.show()
