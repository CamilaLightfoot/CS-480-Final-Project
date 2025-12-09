import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import insightface
from torchvision import transforms
from pathlib import Path
import sys

# Add external/AdaFace to sys.path so we can import AdaFace's modules.
ADA_FACE_ROOT = Path(__file__).resolve().parent.parent / "external" / "AdaFace"
if ADA_FACE_ROOT.exists():
    sys.path.append(str(ADA_FACE_ROOT))

# AdaFace imports – these come from the AdaFace repo.
try:
    from inference import load_pretrained_model, to_input
    from face_alignment import align
except ImportError:
    raise ImportError("AdaFaceModel: cannot import 'load_pretrained_model'. Install AdaFace repo and make sure it's on PYTHONPATH.")


class BaseFaceModel(ABC):
    """Abstract base for all face embedding models."""

    @abstractmethod
    def get_embedding(self, img_path: str) -> np.ndarray:
        """Return an L2-normalized embedding for the face in img_path."""
        raise NotImplementedError

class FaceNetModel(BaseFaceModel):
    """
    FaceNet (InceptionResnetV1 from facenet-pytorch).

    IMPORTANT:
    - For already aligned datasets like LFW-deepfunneled, we SKIP MTCNN and just resize the image to 160x160.
    - For raw/unprocessed images, we still use MTCNN for detection + alignment.
    """

    def __init__(self, device: str = "cuda"):
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # MTCNN for detection + alignment (used only for non-prealigned data).
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            post_process=True,
            device=self.device,
        )

        # Pretrained FaceNet backbone.
        self.model: nn.Module = (
            InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        )

        # Simple transform for pre-aligned faces (e.g., LFW-deepfunneled).
        self.prealigned_transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                # InceptionResnetV1 expects roughly [-1, 1].
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _align_with_mtcnn(self, img: Image.Image) -> Optional[torch.Tensor]:
        """
        Run MTCNN on a PIL image and return aligned face tensor (3, 160, 160) in [0,1], or None if detection fails.
        """
        return self.mtcnn(img)

    def _use_prealigned(self, img_path: str) -> bool:
        """
        Decide if we should skip MTCNN and treat the image as already aligned.

        We skip detection for:
        - Original LFW deepfunneled images
        - Any robustness-augmented images stored in robust_tmp/
        """

        p = img_path.replace("\\", "/").lower()

        return (
            "lfw-deepfunneled" in p
            or "/lfw/" in p
            or "robust_tmp" in p
        )

    def get_embedding(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")

        # 1) Pre-aligned datasets: skip detection and just resize/normalize.
        if self._use_prealigned(img_path):
            img_tensor = self.prealigned_transform(img).unsqueeze(0).to(self.device)

        # 2) Other images: use MTCNN for detection + alignment.
        else:
            aligned = self._align_with_mtcnn(img)
            if aligned is None:
                raise RuntimeError(f"FaceNet: no face detected in {img_path}")
            img_tensor = aligned.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(img_tensor)

        emb = emb.cpu().numpy().reshape(-1)

        # L2 normalize.
        norm = np.linalg.norm(emb) + 1e-10
        emb = emb / norm
        return emb.astype(np.float32)

class ArcFaceModel(BaseFaceModel):
    """
    ArcFace embedding using InsightFace's FaceAnalysis pipeline.
    Uses RetinaFace-style detection + ArcFace recognition internally.
    """

    def __init__(self, device: str = "cuda"):
        if torch.cuda.is_available() and device.startswith("cuda"):
            ctx_id = 0   # CUDA (GPU) Mode
        else:
            ctx_id = -1  # CPU mode

        # This 'buffalo_l' pack includes an ArcFace-style recognition model.
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_embedding(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"ArcFace: cannot read image {img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise RuntimeError(f"ArcFace: no face detected in {img_path}")

        # Take the first detected face (you could later add a strategy if multiple).
        face = faces[0]

        # InsightFace already returns a normalized embedding (normed_embedding).
        if getattr(face, "normed_embedding", None) is not None:
            emb = face.normed_embedding
        else:
            emb = face.embedding
            norm = np.linalg.norm(emb) + 1e-10
            emb = emb / norm

        return np.asarray(emb, dtype=np.float32)

class AdaFaceModel(BaseFaceModel):
    """
    AdaFace embedding using the official AdaFace repo.

    Behavior:
      * For pre-aligned datasets (LFW-deepfunneled, robust_tmp), we SKIP MTCNN and just resize the image to 112x112.
      * For other images, we use facenet-pytorch MTCNN to detect & align.
      * We convert RGB -> BGR and normalize with mean=0.5, std=0.5 as in the AdaFace README.
    """

    def __init__(self, device: str = "cuda", arch: str = "ir_50"):
        if load_pretrained_model is None:
            raise ImportError("AdaFaceModel: cannot import 'load_pretrained_model'. Install AdaFace repo and make sure it's on PYTHONPATH.")

        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ckpt_path = ADA_FACE_ROOT / "pretrained" / f"adaface_{arch}_ms1mv2.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"AdaFace checkpoint not found at: {ckpt_path}")

        # Load raw state_dict from the checkpoint file.
        state = torch.load(str(ckpt_path), map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]

        # Build AdaFace backbone.
        self.model: nn.Module = load_pretrained_model(arch).to(self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # MTCNN for non-prealigned data.
        self.mtcnn = MTCNN(
            image_size=112,
            margin=0,
            min_face_size=20,
            post_process=True,
            device=self.device,
        )

    def _use_prealigned(self, img_path: str) -> bool:
        """
        Decide if this image should be treated as already aligned.

        We skip MTCNN for:
          - LFW deepfunneled images
          - robustness temp images in robust_tmp/
        """
        p = img_path.replace("\\", "/").lower()
        return ("lfw-deepfunneled" in p) or ("robust_tmp" in p)

    def _load_rgb_01_prealigned(self, img_path: str) -> np.ndarray:
        """
        Load a pre-aligned face image as RGB float32 in [0,1], HWC, 112x112.
        """
        img = Image.open(img_path).convert("RGB")
        img = img.resize((112, 112), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0  # HWC, RGB, [0,1]
        return arr

    def _load_rgb_01_mtcnn(self, img_path: str) -> np.ndarray:
        """
        Run MTCNN on the image and return aligned RGB float32 in [0,1], HWC.
        """
        img = Image.open(img_path).convert("RGB")
        aligned = self.mtcnn(img)  # C,H,W in [0,1] or None
        if aligned is None:
            raise RuntimeError(f"AdaFace: no face detected in {img_path}")
        arr = aligned.permute(1, 2, 0).cpu().numpy()  # HWC, RGB, [0,1]
        return arr

    def get_embedding(self, img_path: str) -> np.ndarray:
        # 1) Load image as RGB [0,1], HWC, 112x112.
        if self._use_prealigned(img_path):
            rgb = self._load_rgb_01_prealigned(img_path)
        else:
            rgb = self._load_rgb_01_mtcnn(img_path)

        # 2) Convert RGB -> BGR and normalize (mean=0.5, std=0.5).
        bgr = rgb[:, :, ::-1]                # RGB -> BGR
        bgr = (bgr - 0.5) / 0.5              # normalize

        img = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0).float()
        img = img.to(self.device)            # (1, 3, 112, 112)

        with torch.no_grad():
            feat, _ = self.model(img)        # (1, 512)

        emb = feat.cpu().numpy().reshape(-1)
        norm = np.linalg.norm(emb) + 1e-10
        emb = emb / norm
        return emb.astype(np.float32)

def get_model(name: str, device: str = "cuda") -> BaseFaceModel:
    """
    Factory to obtain a face embedding model by name.

    name ∈ {"facenet", "arcface", "adaface"} (case-insensitive)
    """
    name_lower = name.lower()
    if name_lower == "facenet":
        return FaceNetModel(device=device)
    elif name_lower == "arcface":
        return ArcFaceModel(device=device)
    elif name_lower == "adaface":
        return AdaFaceModel(device=device)
    else:
        raise ValueError(f"Unknown model name '{name}'. "
                         f"Supported: facenet, arcface, adaface.")