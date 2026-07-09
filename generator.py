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
    VRAM_GB = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._g_dino = None
        self._g_dino_processor = None
        self._sam = None
        self._sam_processor = None

    def _auto_download(self) -> None:
        from huggingface_hub import snapshot_download

        # Download IP2P model to root of model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.MODEL_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )
        # Remove corrupt safetensors files so loader falls back to .bin
        for safetensors_file in self.model_dir.rglob("*.safetensors"):
            bin_file = safetensors_file.with_suffix(".bin")
            if bin_file.exists():
                ratio = safetensors_file.stat().st_size / bin_file.stat().st_size
                if ratio < 0.5:
                    print(f"[_auto_download] Removing corrupt safetensors: {safetensors_file.name} ({safetensors_file.stat().st_size} bytes, {ratio:.1%} of .bin)")
                    safetensors_file.unlink()

        # Download G-DINO + SAM to subdirectories
        repos = {
            "grounding_dino": "IDEA-Research/grounding-dino-base",
            "sam": "facebook/sam-vit-base",
        }
        for subdir, repo_id in repos.items():
            target_dir = self.model_dir / subdir
            if target_dir.exists() and any(target_dir.iterdir()):
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
            )

    def is_downloaded(self) -> bool:
        ip2p_ok = (self.model_dir / "model_index.json").exists()
        gd_ok = (self.model_dir / "grounding_dino").exists() and any((self.model_dir / "grounding_dino").iterdir())
        sam_ok = (self.model_dir / "sam").exists() and any((self.model_dir / "sam").iterdir())
        return ip2p_ok and gd_ok and sam_ok

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

        self._report(cb, 40, "Loading GroundingDINO...")
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        gd_path = str(self.model_dir / "grounding_dino")
        self._g_dino = GroundingDinoForObjectDetection.from_pretrained(gd_path, local_files_only=True).to(device)
        self._g_dino_processor = GroundingDinoProcessor.from_pretrained(gd_path, local_files_only=True)

        self._report(cb, 65, "Loading SAM...")
        from transformers import SamModel, SamProcessor

        sam_path = str(self.model_dir / "sam")
        self._sam = SamModel.from_pretrained(sam_path, local_files_only=True).to(device)
        self._sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True)

        self._report(cb, 100, "Ready")

    def unload(self) -> None:
        self._pipe = None
        self._g_dino = None
        self._g_dino_processor = None
        self._sam = None
        self._sam_processor = None
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
        image_guidance = float(params.get("image_guidance_scale", 2.5))
        num_images = int(params.get("num_images", 1))
        seed = int(params.get("seed", 0))
        auto_mask = params.get("auto_mask", True)

        src = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = src.size

        # ---- Step 1: Auto-mask (if enabled) ----
        mask_img = None
        if auto_mask:
            boxes = self._detect(src, prompt, cancel_event, threshold=0.2)
            if not boxes:
                target = self._extract_target(prompt)
                if target and target != prompt:
                    boxes = self._detect(src, target, cancel_event, threshold=0.15)

            if boxes:
                self._report(progress_cb, 10, "Segmenting region...")
                mask_np = self._segment(src, boxes[0], cancel_event)
                mask_img = PILImage.fromarray(mask_np, mode="L")
                # Soften mask edge for natural compositing
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
            else:
                self._report(progress_cb, 10, "No mask — full image edit")

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

    @staticmethod
    def _extract_target(prompt: str) -> str:
        prompt = prompt.strip().lower()
        skip_words = [
            "the", "my", "this", "that", "his", "her", "their", "your", "our", "its", "a", "an",
        ]
        skip_words.sort(key=len, reverse=True)
        skip = r"(?:" + "|".join(skip_words) + r")"
        patterns = [
            rf"(?:make|turn|change|paint|color|replace)\s+{skip}\s+(\w+)",
            rf"(?:make|turn|change|paint|color|replace)\s+(\w+)",
            rf"(\w+)\s+(?:into|to|as)\s+\w+",
        ]
        for pat in patterns:
            m = re.search(pat, prompt)
            if m:
                return m.group(1)
        return prompt.split()[0] if prompt else ""

    def _detect(
        self, image: PILImage.Image, text: str, cancel_event: threading.Event,
        threshold: float = 0.2,
    ) -> list:
        self._check_cancelled(cancel_event)
        inputs = self._g_dino_processor(
            images=image, text=text, return_tensors="pt"
        ).to(self._g_dino.device)
        with torch.inference_mode():
            outputs = self._g_dino(**inputs)

        h, w = image.size[1], image.size[0]
        target_size = torch.tensor([[h, w]], device=self._g_dino.device)
        results = self._g_dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=target_size,
        )

        boxes = results[0].get("boxes")
        scores = results[0].get("scores")
        if boxes is None or len(boxes) == 0:
            return []
        idx = scores.argsort(descending=True)
        return [boxes[i].tolist() for i in idx]

    def _segment(
        self, image: PILImage.Image, box: list, cancel_event: threading.Event
    ) -> np.ndarray:
        self._check_cancelled(cancel_event)
        inputs = self._sam_processor(
            images=image, input_boxes=[[box]], return_tensors="pt"
        ).to(self._sam.device)

        with torch.inference_mode():
            outputs = self._sam(**inputs)

        masks = self._sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        best_idx = outputs.iou_scores[0].argmax().item()
        mask_np = (masks[0][0, best_idx].numpy() * 255).astype(np.uint8)
        return mask_np

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
