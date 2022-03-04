import torch

def print_stats(name, arr):
    mean = arr.mean()
    med = arr.median()
    std = arr.std()
    print(f"Arr: {name}\nMean: {mean}, Std: {std}, Median: {med}")

eng_vectors = torch.tensor(torch.load("experiments/eng_reconstruct/train.z.pt"))
spa_vectors = torch.tensor(torch.load("experiments/spa_reconstruct/train.z.pt"))

# Lienar
diff = eng_vectors - spa_vectors
dist = diff.norm(dim=1)
print(f"Diff: {diff} ({diff.shape})\nDist: {dist} ({dist.shape})")

print_stats("Dist", dist)

# Radial
dots = eng_vectors @ spa_vectors.t()
print(f"Dot product: {dots}")

cos = torch.diagonal(dots) / (spa_vectors.norm(dim=1) * eng_vectors.norm(dim=1))
print(f"Cosine: {cos}")

rad = torch.acos(cos)
print(f"Radian: {rad}")

deg = torch.rad2deg(rad)
print(f"Degree: {deg}")
print_stats("Degree", deg)
