import gc
import io
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter

from services.generators.base import BaseGenerator, GenerationCancelled


class InstructPix2PixGenerator(BaseGenerator):
    MODEL_ID = "timbrooks/instruct-pix2pix"
    DISPLAY_NAME = "InstructPix2Pix Edit"
    VRAM_GB = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clipseg = None
        self._clipseg_processor = None

    def _auto_download(self) -> None:
        from huggingface_hub import snapshot_download

        # Download IP2P model
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.MODEL_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )
        for f in self.model_dir.rglob("*.safetensors"):
            bin_file = f.with_suffix(".bin")
            if bin_file.exists():
                ratio = f.stat().st_size / bin_file.stat().st_size
                if ratio < 0.5:
                    f.unlink()

        # Download CLIPSeg (text-to-mask model)
        clipseg_dir = self.model_dir / "clipseg"
        clipseg_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="CIDAS/clipseg-rd64-refined",
            local_dir=str(clipseg_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )

    def is_downloaded(self) -> bool:
        ip2p_ok = (self.model_dir / "model_index.json").exists()
        clipseg_ok = (self.model_dir / "clipseg").exists() and any((self.model_dir / "clipseg").iterdir())
        return ip2p_ok and clipseg_ok

    def load(self) -> None:
        cb = getattr(self, "_progress", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._report(cb, 5, "Loading InstructPix2Pix...")
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self._pipe.enable_attention_slicing()
        self._pipe.vae.enable_slicing()
        self._pipe.vae.enable_tiling()
        self._pipe.to(device)

        self._report(cb, 50, "Loading CLIPSeg...")
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        clipseg_path = str(self.model_dir / "clipseg")
        self._clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_path, local_files_only=True).to(device)
        self._clipseg_processor = AutoProcessor.from_pretrained(clipseg_path, local_files_only=True)

        self._report(cb, 100, "Ready")

    def unload(self) -> None:
        self._pipe = None
        self._clipseg = None
        self._clipseg_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Union[Path, list[Path]]:
        self._progress = progress_cb
        self._cancel = cancel_event

        prompt = params.get("prompt", "")
        if not prompt.strip():
            raise ValueError("Edit instruction cannot be empty.")

        steps = int(params.get("steps", 30))
        guidance_scale = float(params.get("guidance_scale", 7.5))
        image_guidance = float(params.get("image_guidance_scale", 1.5))
        num_images = int(params.get("num_images", 1))
        seed = int(params.get("seed", 0))

        auto_mask_raw = params.get("auto_mask", True)
        if isinstance(auto_mask_raw, str):
            auto_mask = auto_mask_raw.lower() in ("true", "1", "yes", "on")
        elif isinstance(auto_mask_raw, bool):
            auto_mask = auto_mask_raw
        else:
            auto_mask = bool(auto_mask_raw)

        manual_target = params.get("target", "").strip()

        src = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = src.size

        # ---- Step 1: Auto-mask via CLIPSeg ----
        mask_img = None
        if auto_mask:
            search_text = manual_target or prompt
            self._report(progress_cb, 5, f"Generating mask for '{search_text}'...")
            mask_np = self._text_to_mask(src, search_text, cancel_event)
            mask_img = PILImage.fromarray(mask_np, mode="L")
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))

        # ---- Step 2: Run IP2P (global edit) ----
        self._report(progress_cb, 15, "Editing...")

        gen = None
        if seed != 0 and torch.cuda.is_available():
            gen = torch.Generator(device="cuda").manual_seed(seed)

        def step_callback(pipe, step, timestep, callback_kwargs):
            self._check_cancelled(cancel_event)
            pct = 15 + int((step / steps) * 70)
            self._report(progress_cb, pct, f"Step {step}/{steps}")
            return callback_kwargs

        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                image=src,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance,
                num_images_per_prompt=num_images,
                generator=gen,
                callback_on_step_end=step_callback,
                output_type="pil",
            )

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 90, "Compositing...")

        # ---- Step 3: Composite through mask ----
        paths = []
        for i, img in enumerate(result.images):
            if mask_img is not None:
                m = mask_img.resize(img.size, PILImage.LANCZOS)
                s = src.resize(img.size, PILImage.LANCZOS)
                img = PILImage.composite(img, s, m)
                img = img.resize(orig_size, PILImage.LANCZOS)

            filename = f"ip2p_edit_{self._timestamp()}_{i}.png"
            out_path = self.outputs_dir / filename
            img.save(str(out_path), "PNG")
            paths.append(out_path)

        self._report(progress_cb, 100, "Done")

        if len(paths) == 1:
            return paths[0]
        return paths

    def _text_to_mask(
        self, image: PILImage.Image, text: str, cancel_event: threading.Event
    ) -> np.ndarray:
        self._check_cancelled(cancel_event)
        inputs = self._clipseg_processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self._clipseg.device)

        with torch.inference_mode():
            outputs = self._clipseg(**inputs)

        # logits shape: (1, 1, H, W) — squeeze and sigmoid to [0,1]
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        # Threshold: 0.3 gives a good balance
        mask = (probs > 0.3).astype(np.uint8) * 255
        # Resize back to original image size (CLIPSeg may output at lower res)
        mask_pil = PILImage.fromarray(mask, mode="L")
        mask_pil = mask_pil.resize(image.size, PILImage.LANCZOS)
        return np.array(mask_pil)

    def _report(self, cb, pct, msg):
        if cb:
            cb(pct, msg)

    def _check_cancelled(self, cancel_event):
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

    @staticmethod
    def _timestamp() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
