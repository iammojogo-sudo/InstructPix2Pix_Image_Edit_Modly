# InstructPix2Pix Edit — Modly Extension

Instruction-based image editing using [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix).

Provide an image and a text instruction — *"make it an oil painting"*, *"turn the dog into a cat"* — and get an edited image back.

## Features

- Image-to-image editing via text instruction
- Dual guidance: control how much to follow the instruction vs preserve the original
- Fits 4-6 GB VRAM (fp16)
- 512×512 output resolution

## Models

| Model | Source | Parameters |
|-------|--------|------------|
| InstructPix2Pix | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) | ~1B (diffusers format) |

## License

MIT
