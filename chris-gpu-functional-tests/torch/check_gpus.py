import torch
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def check_gpus():
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    for i in range(gpu_count):
        try:
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name} - Available for PyTorch")
        except Exception as e:
            print(f"GPU {i}: Error accessing device - {e}")

    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use the GPU(s).")
    else:
        print("CUDA is not available. PyTorch will use CPU.")

if __name__ == "__main__":
    check_gpus()
