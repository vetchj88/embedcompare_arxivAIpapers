"""Vision encoder utilities for figure thumbnails.

The encoder is intentionally lightweight: it loads a CLIP vision backbone via
Hugging Face Transformers when available and falls back to deterministic
placeholders when the dependency stack is missing.  This keeps the module usable
in offline evaluation environments while enabling full fidelity embeddings when
PyTorch and Transformers are installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency resolution
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - executed only when deps missing
    torch = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

from PIL import Image


@dataclass
class VisionBatch:
    """Container for returning embeddings and ancillary data."""

    embeddings: "torch.Tensor"
    image_sizes: List[Tuple[int, int]]


class FigureEncoder:
    """Encode figure thumbnails using a CLIP vision backbone."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> None:
        if torch is None or CLIPModel is None or CLIPProcessor is None:  # pragma: no cover - depends on env
            raise RuntimeError(
                "FigureEncoder requires PyTorch and transformers. Install them to "
                "enable visual embedding support."
            )

        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _prepare_images(self, images: Sequence[Union[str, Image.Image]]) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
        pil_images: List[Image.Image] = []
        sizes: List[Tuple[int, int]] = []

        for image in images:
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                pil_image = Image.open(image).convert("RGB")
            sizes.append(pil_image.size)
            pil_images.append(pil_image)

        return pil_images, sizes

    def encode(self, images: Sequence[Union[str, Image.Image]], batch_size: int = 8) -> VisionBatch:
        """Return CLIP embeddings for ``images``.

        Parameters
        ----------
        images:
            Iterable of ``PIL.Image`` objects or paths to image files.
        batch_size:
            Number of images to encode per forward pass.
        """

        if torch is None:  # pragma: no cover - safety guard
            raise RuntimeError("PyTorch is required to encode figure thumbnails.")

        pil_images, sizes = self._prepare_images(images)
        embeddings: List["torch.Tensor"] = []

        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_outputs = self.model.get_image_features(**inputs)
            embeddings.append(vision_outputs)

        stacked = torch.cat(embeddings, dim=0)
        # Normalise embeddings so cosine similarity corresponds to CLIP defaults
        stacked = stacked / stacked.norm(p=2, dim=-1, keepdim=True)
        return VisionBatch(embeddings=stacked, image_sizes=sizes)


__all__ = ["FigureEncoder", "VisionBatch"]
