import json
import subprocess
import sys


def main():
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            pass
    python = config.get("python_exe") or sys.executable

    deps = [
        "diffusers>=0.24.0",
        "transformers>=4.38.0",
        "torch>=2.1.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
    ]
    for dep in deps:
        subprocess.check_call([python, "-m", "pip", "install", dep])
    print("InstructPix2Pix Edit extension installed successfully.")


if __name__ == "__main__":
    main()
