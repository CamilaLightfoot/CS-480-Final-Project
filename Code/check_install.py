"""
Ensure we have all the project dependencies.

Run with:
    python check_install.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")
import sys

print("Python version:", sys.version)

# ---- Torch & CUDA ----
try:
    import torch

    print("\n[torch]")
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU only.")
except Exception as e:
    print("[ERROR] torch import failed:", e)

# ---- torchvision ----
try:
    import torchvision

    print("\n[torchvision]")
    print("torchvision version:", torchvision.__version__)
except Exception as e:
    print("[ERROR] torchvision import failed:", e)

# ---- facenet-pytorch (FaceNet + MTCNN) ----
try:
    from facenet_pytorch import InceptionResnetV1, MTCNN

    print("\n[facenet-pytorch]")
    print("facenet-pytorch imported successfully.")

    # Simple model instantiation test (no forward pass on GPU yet)
    _mtcnn = MTCNN(image_size=160, margin=0)
    _facenet = InceptionResnetV1(pretrained="vggface2").eval()
    print("MTCNN + InceptionResnetV1 instantiated successfully.")
except Exception as e:
    print("[ERROR] facenet-pytorch import or model init failed:", e)

# ---- insightface (ArcFace / RetinaFace) ----
try:
    import insightface

    print("\n[insightface]")
    print("insightface version:", getattr(insightface, '__version__', 'unknown'))

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    print("InsightFace FaceAnalysis instantiated successfully.")
except Exception as e:
    print("[ERROR] insightface or FaceAnalysis failed:", e)

# ---- AdaFace ----
try:
    import adaface

    print("\n[adaface]")
    print("adaface imported successfully.")
    # We won't instantiate a model here (API may change); this just checks installation.
except Exception as e:
    print("[ERROR] adaface import failed:", e)

# ---- Other libs ----
for name in ["numpy", "scipy", "PIL", "matplotlib", "sklearn", "tqdm", "umap"]:
    try:
        if name == "PIL":
            import PIL  # noqa: F401
        elif name == "matplotlib":
            import matplotlib  # noqa: F401
        elif name == "sklearn":
            import sklearn  # noqa: F401
        elif name == "umap":
            import umap  # noqa: F401
        else:
            __import__(name)
        print(f"[{name}] imported successfully.")
    except Exception as e:
        print(f"[ERROR] {name} import failed:", e)

print("\n check_install.py completed.")
