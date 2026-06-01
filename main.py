# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""
LocalAI Studio — entry point.
Checks / installs Python dependencies, then launches the GUI.
"""

import sys
import subprocess
import importlib
import os


REQUIRED_PACKAGES = {
    "customtkinter": "customtkinter",
    "requests": "requests",
    "psutil": "psutil",
}

if sys.platform == "darwin":
    OPTIONAL_PACKAGES = {
        "onnxruntime": "onnxruntime",
        "coremltools": "coremltools",
        "optimum": "optimum[onnxruntime]",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub>=0.34.0,<1.0",
    }
else:
    OPTIONAL_PACKAGES = {
        "onnxruntime_directml": "onnxruntime-directml",
        "onnxruntime_genai": "onnxruntime-genai-directml",
        "optimum": "optimum[onnxruntime]",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub>=0.34.0,<1.0",
    }


def _ensure(packages: dict, label: str = ""):
    missing = []
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return True

    auto_install = (
        os.environ.get("LOCALAI_AUTO_INSTALL_DEPS") == "1"
        or "--install-deps" in sys.argv
    )
    if auto_install:
        print(f"\n{'=' * 60}")
        print(f"Installing {label or 'required'} packages: {', '.join(missing)}")
        print("=" * 60)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *missing]
        )
        print("Done.\n")
        return True

    print()
    print("[ERROR] Missing required Python packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print()
    print("Run setup.bat first, or install dependencies with:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print()
    print("For developer auto-install only, run with --install-deps or set LOCALAI_AUTO_INSTALL_DEPS=1.")
    return False


def main():
    if not _ensure(REQUIRED_PACKAGES, "required"):
        sys.exit(1)

    from src.app import APP_VERSION
    print()
    print("!" * 68)
    print("!")
    print("!  [!]  DO NOT CLOSE THIS WINDOW  [!]")
    print("!")
    print("!  LocalAI Studio is running in this window.")
    print("!  Closing it will immediately kill the app, cancel any active")
    print("!  downloads, and may corrupt in-progress tasks.")
    print("!")
    print("!  MINIMIZE this window instead - do not close it.")
    print("!")
    print("!" * 68)
    print()
    print(f"LocalAI Studio v{APP_VERSION} starting …")
    if sys.platform == "darwin":
        print("Optional CoreML/ONNX packages can be installed via Settings or:")
        print('  pip install onnxruntime coremltools "optimum[onnxruntime]" transformers "huggingface-hub>=0.34.0,<1.0"')
        print("Utility demos (OCR/speech/embeddings/document AI) also use:")
        print("  pip install torch sentence-transformers soundfile scipy pillow timm sentencepiece librosa backoff hf_xet")
    else:
        print("Optional NPU/DirectML packages can be installed via Settings or:")
        print('  pip install onnxruntime-directml onnxruntime-genai-directml "optimum[onnxruntime]" transformers "huggingface-hub>=0.34.0,<1.0"')
        print("Utility demos (OCR/speech/embeddings/document AI) also use:")
        print("  pip install torch torchvision sentence-transformers soundfile scipy pillow timm sentencepiece librosa backoff accelerate safetensors einops hf_xet")

    from src.app import App
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
