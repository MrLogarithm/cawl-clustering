It is crucial that the model inputs are z-score normalized. If the color channels are not centered around zero, the model will not learn the data.

Cluster these to recover the script inventory: `'vae_mu', 'vae_sigma', 'vae_z'`

## Basic Usage
```
import torch
import matplotlib.pyplot as plt
import numpy as np

from expts import expts
import model

# Create model from scratch:
params = expts[expt]
n_dims = 64
device = "cuda"
model = model.VAECluster(
    expts["txr"], # one of 'vae-self', 'vae-ctx', 'lstm', or 'txr'
    n_dims,
).to(device)

# Or load a pretrained model:
pretrained = torch.load(
    "models/txr_jp.pt",
    weights_only=False,
    map_location=device,
)

# Load data:
data = np.load(
    f"data/jp/test.npz",
)
# Image inputs should be z-score normalized:
images = data['images']
images = (images - images.mean()) / images.std()
# Unsqueeze to create a single color channel:
images = images[:,None,:,:]
images = torch.tensor(images, device=device).float()
# Model inputs should end up with the shape 
# (sequence_len, channels, width, height)

result = pretrained(
    images,
)

index = 7
plt.subplot(1,3,1)
plt.imshow(images[index, 0].detach().cpu())
plt.axis('off')
plt.title("Target")

plt.subplot(1,3,2)
plt.imshow(result["gen_txr"][index,0].detach().cpu())
plt.axis('off')
plt.title("Transformer\nReconstruction")

plt.subplot(1,3,3)
plt.imshow(result["gen_vae_self"][index,0].detach().cpu())
plt.axis('off')
plt.title("VAE\nReconstruction")
plt.show()
```

[](assets/images.png)
