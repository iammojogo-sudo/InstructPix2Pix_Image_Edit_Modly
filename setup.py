"""
InstructPix2Pix Edit — Image editing for Modly
"""
import subprocess
import sys


def main():
    deps = [
        "diffusers>=0.24.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "pillow>=10.0.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
    ]
    for dep in deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    print("InstructPix2Pix Edit extension installed successfully.")


if __name__ == "__main__":
    main()
