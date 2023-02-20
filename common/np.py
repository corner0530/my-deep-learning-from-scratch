"""NumPyのGPU対応モードを切り替える

Attributes:
    GPU (bool): GPUモードの有効/無効
    np (module): GPUに対応するときはCuPy，しない場合はNumPy
"""

GPU = False

if GPU:
    import cupy as np

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print("\033[92m" + "-" * 60 + "\033[0m")
    print(" " * 23 + "\033[92mGPU Mode (cupy)\033[0m")
    print("\033[92m" + "-" * 60 + "\033[0m\n")
else:
    import numpy as np
