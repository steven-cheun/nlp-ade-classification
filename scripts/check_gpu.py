import sys
import torch


# Minimum requirements for this project
MIN_VRAM_GB = 8
MIN_COMPUTE_CAPABILITY = (7, 0)  # Volta minimum for fp16 support


def check_gpu():
    print("=" * 50)
    print("GPU / CUDA DIAGNOSTICS")
    print("=" * 50)

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("\n[ERROR] No CUDA-capable GPU detected.")
        print("This project requires an NVIDIA GPU with CUDA support.")
        print("Training will not proceed on CPU — BioBERT fine-tuning requires a GPU.")
        sys.exit(1)

    print(f"CUDA version  : {torch.version.cuda}")
    print(f"cuDNN version : {torch.backends.cudnn.version()}")
    print(f"GPU count     : {torch.cuda.device_count()}")
    print()

    all_ok = True

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024 ** 3)
        cc = (props.major, props.minor)

        print(f"  GPU {i}: {props.name}")
        print(f"    Compute capability : {props.major}.{props.minor}", end="")

        # Friendly architecture label
        if cc >= (12, 0):
            print("  (Blackwell)")
        elif cc >= (9, 0):
            print("  (Hopper)")
        elif cc >= (8, 0):
            print("  (Ampere)")
        elif cc >= (7, 5):
            print("  (Turing)")
        elif cc >= (7, 0):
            print("  (Volta)")
        else:
            print(f"  (older architecture — may not support fp16)")

        # Check compute capability minimum
        if cc < MIN_COMPUTE_CAPABILITY:
            print(f"    [WARNING] Compute capability {cc} is below minimum "
                  f"{MIN_COMPUTE_CAPABILITY} required for fp16 training.")
            all_ok = False
        else:
            print(f"    Compute capability : OK (>= {MIN_COMPUTE_CAPABILITY})")

        # Check VRAM minimum
        print(f"    Total VRAM         : {total_mem:.2f} GB", end="")
        if total_mem < MIN_VRAM_GB:
            print(f"  [WARNING] Less than {MIN_VRAM_GB}GB — BioBERT fine-tuning "
                  f"may run out of memory. Reduce batch size.")
            all_ok = False
        else:
            print(f"  OK (>= {MIN_VRAM_GB}GB)")

        print(f"    Multiprocessors    : {props.multi_processor_count}")
        print()

    # Smoke test
    print("Smoke test — matmul on GPU...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"  matmul OK on {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  [ERROR] matmul FAILED: {e}")
        all_ok = False

    # fp16 smoke test
    print("Smoke test — fp16 on GPU...")
    try:
        x = torch.randn(512, 512, dtype=torch.float16).cuda()
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"  fp16 OK on {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  [WARNING] fp16 FAILED: {e}")
        print("  Training will fall back to fp32 (slower).")

    print()
    print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic : {torch.backends.cudnn.deterministic}")
    print("=" * 50)

    if all_ok:
        print("\n[OK] GPU is ready for training.")
    else:
        print("\n[WARNING] GPU checks completed with warnings. "
              "Review above before running training.")

    return all_ok


if __name__ == "__main__":
    ok = check_gpu()
    sys.exit(0 if ok else 1)
