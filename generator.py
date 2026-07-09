import gc
import io
import threading
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image as PILImage

from services.generators.base import BaseGenerator, GenerationCancelled


class InstructPix2PixGenerator(BaseGenerator):
    MODEL_ID = "timbrooks/instruct-pix2pix"
    DISPLAY_NAME = "InstructPix2Pix Edit"
    VRAM_GB = 4

    # ---- download (mirrors the working extensions in this install) ----
    def _auto_download(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Download the FULL diffusers repo into the LOCAL model_dir.
        # We pass local_dir=model_dir so nothing touches the poisoned
        # ~/.cache/huggingface hub cache. snapshot_download writes
        # the component subfolders (unet/, vae/, text_encoder/, ...) here.
        snapshot_download(
            repo_id=self.MODEL_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"],
        )

    def load(self) -> None:
        cb = getattr(self, "_progress", None)
        self._report(cb, 5, "Loading InstructPix2Pix...")

        from diffusers import StableDiffusionInstructPix2PixPipeline

        self._report(cb, 40, "Loading pipeline...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )

        self._report(cb, 70, "Optimizing...")
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        self._model = pipe
        self._report(cb, 100, "Ready")

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Union[str, list[str]]:
        self._progress = progress_cb
        self._cancel = cancel_event

        pipe = self._model
        if pipe is None:
            self.load()

        prompt = params.get("prompt", "")
        if not prompt.strip():
            raise ValueError("Edit instruction cannot be empty.")

        steps = int(params.get("steps", 30))
        guidance_scale = float(params.get("guidance_scale", 7.5))
        image_guidance = float(params.get("image_guidance_scale", 2.5))
        num_images = int(params.get("num_images", 1))
        seed = int(params.get("seed", 0))

        gen = None
        if seed != 0 and torch.cuda.is_available():
            gen = torch.Generator(device="cuda").manual_seed(seed)

        init_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        self._report(progress_cb, 10, "Editing...")

        def step_callback(pipe, step, timestep, callback_kwargs):
            self._check_cancelled(cancel_event)
            pct = 10 + int((step / steps) * 80)
            self._report(progress_cb, pct, f"Step {step}/{steps}")
            return callback_kwargs

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=init_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance,
                num_images_per_prompt=num_images,
                generator=gen,
                callback_on_step_end=step_callback,
                output_type="pil",
            )
        images = result.images

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 95, "Saving...")

        paths = []
        for i, img in enumerate(images):
            filename = f"ip2p_edit_{self._timestamp()}_{i}.png"
            out_path = self.outputs_dir / filename
            img.save(str(out_path), "PNG")
            paths.append(out_path)

        self._report(progress_cb, 100, "Done")

        if len(paths) == 1:
            return paths[0]
        return paths

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
