import gc
import io
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage, ImageFilter

from services.generators.base import BaseGenerator, GenerationCancelled


class InstructPix2PixGenerator(BaseGenerator):
    MODEL_ID = "timbrooks/instruct-pix2pix"
    DISPLAY_NAME = "InstructPix2Pix Edit"
    VRAM_GB = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pipe = None
        self._sam = None
        self._sam_processor = None
        self._clip = None
        self._clip_processor = None

    def _auto_download(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Download IP2P model
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

        # Download SAM (mask proposal)
        sam_dir = self.model_dir / "sam"
        sam_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="facebook/sam-vit-base",
            local_dir=str(sam_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )

        # Download CLIP (text-to-mask scoring)
        clip_dir = self.model_dir / "clip"
        clip_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            local_dir=str(clip_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )

    def is_downloaded(self) -> bool:
        ip2p_ok = (self.model_dir / "model_index.json").exists()
        sam_ok = (self.model_dir / "sam").exists() and any((self.model_dir / "sam").iterdir())
        clip_ok = (self.model_dir / "clip").exists() and any((self.model_dir / "clip").iterdir())
        return ip2p_ok and sam_ok and clip_ok

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

        self._report(cb, 30, "Loading SAM...")
        from transformers import SamModel, SamProcessor

        sam_path = str(self.model_dir / "sam")
        self._sam = SamModel.from_pretrained(sam_path, local_files_only=True).to(device)
        self._sam_processor = SamProcessor.from_pretrained(sam_path, local_files_only=True)

        self._report(cb, 60, "Loading CLIP...")
        from transformers import CLIPModel, CLIPProcessor

        clip_path = str(self.model_dir / "clip")
        self._clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
        self._clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)

        self._report(cb, 100, "Ready")

    def unload(self) -> None:
        self._pipe = None
        self._sam = None
        self._sam_processor = None
        self._clip = None
        self._clip_processor = None
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

        auto_mask = params.get("auto_mask", "on") == "on"

        src = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = src.size

        # ---- Step 1: Auto-mask via SAM + CLIP ----
        mask_img = None
        if auto_mask:
            target = params.get("target", "").strip() or self._extract_target(prompt)
            self._report(progress_cb, 5, f"Searching for '{target}'...")
            mask_np = self._find_mask(src, target, cancel_event)
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

    def _find_mask(
        self, image: PILImage.Image, target: str, cancel_event: threading.Event
    ) -> np.ndarray:
        self._check_cancelled(cancel_event)
        device = self._sam.device
        h, w = image.height, image.width
        clip_size = 224

        # ---- 1. Preprocess image (keeps size info for post-processing) ----
        inputs = self._sam_processor(image, return_tensors="pt").to(device)
        original_sizes = inputs["original_sizes"].cpu()
        reshaped_input_sizes = inputs["reshaped_input_sizes"].cpu()

        with torch.inference_mode():
            image_embeddings = self._sam.vision_encoder(inputs["pixel_values"]).last_hidden_state

        # ---- 2. Grid point prompts (in original coordinates, scaled by processor) ----
        step = max(32, min(h, w) // 32)
        ys = torch.arange(step // 2, h, step, device=device)
        xs = torch.arange(step // 2, w, step, device=device)
        grid_ys, grid_xs = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([grid_xs.flatten(), grid_ys.flatten()], dim=1).float()

        if len(points) == 0:
            return np.ones((h, w), dtype=np.uint8) * 255

        # ---- 3. Generate mask proposals ----
        all_masks = []
        all_scores = []
        batch_size = 64

        for i in range(0, len(points), batch_size):
            self._check_cancelled(cancel_event)
            batch_pts = points[i:i+batch_size]

            # Use processor to scale points from original coords to SAM input space
            proc = self._sam_processor(image, input_points=[batch_pts.tolist()], return_tensors="pt").to(device)
            scaled_pts = proc["input_points"][0].unsqueeze(1)  # [B, 1, 2]
            batch_lbls = torch.ones(scaled_pts.shape[0], 1, device=device)

            with torch.inference_mode():
                outputs = self._sam(
                    pixel_values=None,
                    input_points=scaled_pts,
                    input_labels=batch_lbls,
                    image_embeddings=image_embeddings.expand(scaled_pts.shape[0], -1, -1, -1),
                    multimask_output=True,
                )
            all_masks.append(outputs.pred_masks.cpu().float())
            all_scores.append(outputs.iou_scores.cpu())

        masks = torch.cat(all_masks, dim=0)
        iou_scores = torch.cat(all_scores, dim=0)

        N, M, Hm, Wm = masks.shape
        masks = masks.view(N * M, Hm, Wm)
        iou_scores = iou_scores.view(N * M)

        # ---- 4. Filter by SAM's predicted IoU ----
        keep = iou_scores > 0.3
        if keep.any():
            masks = masks[keep]
        if len(masks) == 0:
            return np.ones((h, w), dtype=np.uint8) * 255

        # ---- 5. CLIP text encoding (once) ----
        text_inputs = self._clip_processor(text=[target], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            text_features = self._clip.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        # ---- 6. Score each mask with CLIP ----
        image_clip = image.resize((clip_size, clip_size), PILImage.LANCZOS)
        blurred_clip = image_clip.filter(ImageFilter.GaussianBlur(radius=5))

        best_mask_logits = None
        best_score = -1.0
        clip_batch = 32

        for i in range(0, len(masks), clip_batch):
            self._check_cancelled(cancel_event)
            batch_logits = masks[i:i+clip_batch]

            crops = []
            for j in range(len(batch_logits)):
                prob = torch.sigmoid(batch_logits[j])
                prob = F.interpolate(prob[None, None], size=(clip_size, clip_size), mode="bilinear").squeeze()
                msk_pil = PILImage.fromarray((prob.numpy() * 255).astype(np.uint8), mode="L")
                crops.append(PILImage.composite(image_clip, blurred_clip, msk_pil))

            img_inputs = self._clip_processor(images=crops, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                img_features = self._clip.get_image_features(**img_inputs)
            img_features = F.normalize(img_features, dim=-1)

            clip_scores = (img_features @ text_features.T).squeeze(-1)
            batch_best = clip_scores.argmax().item()
            if clip_scores[batch_best] > best_score:
                best_score = clip_scores[batch_best].item()
                best_mask_logits = batch_logits[batch_best].cpu()

        if best_mask_logits is None:
            return np.ones((h, w), dtype=np.uint8) * 255

        # ---- 7. Post-process mask to original resolution (handles padding/crop) ----
        best_mask_batch = best_mask_logits.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
        masks_pp = self._sam_processor.image_processor.post_process_masks(
            best_mask_batch, original_sizes, reshaped_input_sizes
        )
        mask_np = torch.sigmoid(masks_pp[0][0]).numpy()
        return (mask_np * 255).astype(np.uint8)

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
