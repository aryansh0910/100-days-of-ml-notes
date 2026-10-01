# ============================================================
# TENSORS — THE FOUNDATION OF DEEP LEARNING
# ============================================================

# ============================================================
# WHAT IS IT?
# A Tensor is a mathematical object that generalizes
# scalars, vectors, and matrices to higher dimensions.
# It is essentially a CONTAINER FOR NUMBERS.
#
# Think of tensors as the "language" that Deep Learning
# speaks. Everything in PyTorch — your data, your weights,
# your gradients — is a tensor.
#
# RANK = THE NUMBER OF DIMENSIONS OR THE NUMBER OF AXES
# ============================================================


# ============================================================
# THE HIERARCHY — UNDERSTANDING DIMENSIONS
# ============================================================
#
# RANK 0 — SCALAR (A Single Number)
# ─────────────────────────────────
# Just a single number. No axes, no dimensions.
# Example: The price of one car → ₹5,00,000
# LIKE JUST A SINGLE NUMBER
#
#   tensor(500000)
#
# ──────────────────────────────────────────────────────────
#
# RANK 1 — VECTOR (A List of Numbers)
# ─────────────────────────────────────
# A single list of numbers. One axis.
# Example: A single row in your Car Dekho dataset
#          (Year, KM, Fuel) → [2019, 45000, 1]
# LIKE A LIST PASSED IN A NP ARRAY
#
# IMPORTANT DISTINCTION:
# np.array([1,2,3,4]) → this is a 1D TENSOR
# but it is a 4D VECTOR because it has 4 values
# (4 input features). Every row will have these
# 4 input features.
# TENSOR DIMENSIONS ARE NOT ALWAYS EQUAL TO VECTOR DIMENSIONS!
# A 1D TENSOR IS A COLLECTION OF 0D TENSORS (SCALARS)
#
#   tensor([2019, 45000, 1, 0])
#
# ──────────────────────────────────────────────────────────
#
# RANK 2 — MATRIX (A Table of Numbers)
# ──────────────────────────────────────
# Rows and columns — like a spreadsheet or dataframe.
# Example: Your entire X_train dataframe.
# MATRICES ARE COLLECTIONS OF VECTORS
#
#   tensor([[2019, 45000, 1, 0],
#           [2020, 12000, 0, 1],
#           [2018, 67000, 1, 0]])
#
# ──────────────────────────────────────────────────────────
#
# RANK 3 — 3D TENSOR (A Cube of Numbers)
# ────────────────────────────────────────
# Example: A colored image
# Height × Width × 3 Color Channels (Red, Green, Blue)
# IT IS A COLLECTION OF 2D TENSORS
# Example: 4 matrices of size 3×3 → shape is (4, 3, 3)
#
#   Shape: (224, 224, 3) for a standard image
#          ↑     ↑    ↑
#        Height Width RGB
#
# ──────────────────────────────────────────────────────────
#
# RANK 4 — 4D TENSOR (A Collection of Images)
# ─────────────────────────────────────────────
# A VECTOR OF 3D TENSORS — multiple 3D tensors combined.
# Example: A BATCH of images passed to a neural network
#          50 images combined → shape (50, 224, 224, 3)
#          50 3D TENSORS CONTAINING RGB PIXELS COMBINED
#
#   Shape: (batch_size, height, width, channels)
#          (50, 224, 224, 3)
#
# ──────────────────────────────────────────────────────────
#
# RANK 5 — 5D TENSOR (A Video)
# ──────────────────────────────
# 5D IMAGES COMBINED TOGETHER FORMS A VIDEO
# 50 4D TENSORS OR 50 IMAGES COMBINED TOGETHER
# AND PASSED IN 1 SECOND IN FRONT OF OUR EYE
# WILL LOOK LIKE A VIDEO, NOT AN IMAGE
#
#   Shape: (videos, frames, height, width, channels)
#          (10, 30, 224, 224, 3)
#           ↑    ↑    ↑     ↑    ↑
#        #vids fps  H    W   RGB
# ============================================================


# ============================================================
# DESI ANALOGY — CGPA, IQ, STATE EXAMPLE
# ============================================================
#
# FOR THE ANALOGY WE WILL TAKE 3 INPUT FEATURES IN A MODEL
# I.E CGPA, IQ AND STATE
#
# 1D TENSOR → A SINGLE STUDENT'S INPUT FEATURES
#   CGPA=9, IQ=59, STATE=PB
#   tensor([9, 59, 'PB'])
#   → One student, all their features in a row
#
# 2D TENSOR → COLLECTION OF ALL STUDENTS' INPUT FEATURES
#   tensor([
#       [9,  59, 'PB'],   ← student 1
#       [8,  72, 'UP'],   ← student 2
#       [7,  65, 'MH'],   ← student 3
#   ])
#   → The full dataset / X_train matrix
#
# 3D TENSOR → MULTIPLE BATCHES OF STUDENTS
#   tensor([
#       [[9, 59, 'PB'], [8, 72, 'UP']],   ← batch 1
#       [[7, 65, 'MH'], [9, 80, 'DL']],   ← batch 2
#   ])
#   → When you split your dataset into mini-batches
#     for training, each batch is a 3D tensor
# ============================================================


# ============================================================
# SHIPPING CONTAINER ANALOGY
# ============================================================
#
# Scalar     → One item in a box
# Vector     → A row of items in a box
# Matrix     → A full layer of items in a box
# 3D Tensor  → The whole container filled with layers
# 4D Tensor  → A ship carrying thousands of containers
# 5D Tensor  → A fleet of ships over multiple voyages
# ============================================================


# ============================================================
# WHY NOT JUST USE NUMPY ARRAYS?
# ============================================================
#
# You might think: "This just sounds like a NumPy array."
# You are RIGHT! A tensor IS very similar to an array.
# BUT tensors have 2 SUPERPOWERS that NumPy arrays don't:
#
# SUPERPOWER 1 → GPU ACCELERATION
# ─────────────────────────────────
# NumPy arrays run on your CPU only.
# PyTorch Tensors can run on GPUs.
# GPU has thousands of cores vs CPU's 8-16 cores.
# Result: 100x FASTER for heavy matrix math.
#
# This is why training on GPU takes minutes
# but the same on CPU would take hours.
#
# SUPERPOWER 2 → AUTOMATIC DIFFERENTIATION
# ──────────────────────────────────────────
# Tensors keep track of HOW they were calculated.
# This allows PyTorch to automatically calculate
# GRADIENTS (slopes) during backpropagation.
# Gradients = how the model learns from its mistakes.
# Without this, Neural Networks cannot learn at all.
#
# This is called AUTOGRAD in PyTorch.
# Set requires_grad=True and PyTorch tracks everything.
# ============================================================


# ============================================================
# IMPORTS AND BASIC TENSOR OPERATIONS IN PYTORCH
# ============================================================

import torch
import numpy as np

# ── Creating Tensors ─────────────────────────────────────────

scalar = torch.tensor(500000)                        # Rank 0
vector = torch.tensor([9, 59, 1])                    # Rank 1
matrix = torch.tensor([[9, 59, 1],                   # Rank 2
                        [8, 72, 0],
                        [7, 65, 1]])
cube   = torch.rand(3, 224, 224)                     # Rank 3 → one image (RGB)
batch  = torch.rand(50, 3, 224, 224)                 # Rank 4 → batch of 50 images

# ── Checking Tensor Properties ───────────────────────────────

print("Scalar shape    :", scalar.shape)             # torch.Size([])
print("Vector shape    :", vector.shape)             # torch.Size([3])
print("Matrix shape    :", matrix.shape)             # torch.Size([3, 3])
print("Image shape     :", cube.shape)               # torch.Size([3, 224, 224])
print("Batch shape     :", batch.shape)              # torch.Size([50, 3, 224, 224])

print("\nMatrix ndim     :", matrix.ndim)            # 2  ← rank/number of dimensions
print("Matrix dtype    :", matrix.dtype)             # torch.int64
print("Matrix size     :", matrix.numel())           # 9  ← total number of elements

# ── Moving Tensor to GPU (if available) ──────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device    : {device}")

batch_gpu = batch.to(device)                        # moves tensor to GPU
print("Batch on        :", batch_gpu.device)

# ── Autograd — Automatic Differentiation ─────────────────────

x = torch.tensor(3.0, requires_grad=True)           # track this tensor
y = x ** 2 + 2 * x + 1                             # some operation

y.backward()                                        # compute gradients
print(f"\nGradient dy/dx at x=3 : {x.grad}")       # dy/dx = 2x+2 = 8.0

# ── Converting between NumPy and PyTorch ─────────────────────

np_array   = np.array([1, 2, 3])
tensor     = torch.from_numpy(np_array)             # NumPy → PyTorch
back_to_np = tensor.numpy()                         # PyTorch → NumPy

print(f"\nNumPy array  : {np_array}")
print(f"Tensor       : {tensor}")
print(f"Back to NumPy: {back_to_np}")


# ============================================================
# VISUAL MENTAL MODEL — TENSOR HIERARCHY
# ============================================================
#
#  0D  →  42                           ← a price, a score
#
#  1D  →  [9, 59, 1]                   ← one student's features
#
#  2D  →  [[9, 59, 1],                 ← full dataset / X_train
#           [8, 72, 0],
#           [7, 65, 1]]
#
#  3D  →  [[[R, G, B], ...],           ← one image
#           [[R, G, B], ...],
#           [[R, G, B], ...]]
#           H    W    C
#
#  4D  →  50 × [3D image]              ← batch of images
#
#  5D  →  30 frames × [4D batch]       ← video data
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Tensor = container for numbers, generalization of
#    scalar → vector → matrix → higher dimensions
#
# 2. RANK = number of dimensions / number of axes
#    Rank 0 = Scalar, Rank 1 = Vector,
#    Rank 2 = Matrix, Rank 3+ = Tensor
#
# 3. IMPORTANT: 1D Tensor ≠ 1D Vector
#    np.array([1,2,3,4]) is a 1D Tensor but a 4D Vector
#    (4 input features). Don't confuse tensor rank
#    with vector dimensionality!
#
# 4. Each rank is a collection of the rank below it:
#    1D = collection of 0D (scalars)
#    2D = collection of 1D (vectors)
#    3D = collection of 2D (matrices)
#    4D = collection of 3D (images) → batch of images
#    5D = collection of 4D → video data
#
# 5. Tensors have 2 superpowers over NumPy arrays:
#    → GPU Acceleration (100x faster training)
#    → Autograd (automatic gradient calculation)
#
# 6. In PyTorch:
#    tensor.shape  → dimensions
#    tensor.ndim   → rank
#    tensor.dtype  → data type
#    tensor.to('cuda') → move to GPU
#
# 7. Real world tensor shapes:
#    One image     → (3, 224, 224)
#    Batch images  → (50, 3, 224, 224)
#    Video         → (frames, 3, H, W)
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Tensor Operations
#    How to add, multiply, reshape, and slice tensors.
#    Matrix multiplication (the backbone of neural networks),
#    broadcasting, and how operations flow through
#    the computation graph for backpropagation.
# ============================================================