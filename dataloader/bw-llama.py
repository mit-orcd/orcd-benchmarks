import sys
import torch
from torch.utils.data import Dataset, DataLoader
import time

class LLaMAPTDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Loading {pt_file} ...")
        self.data = torch.load(pt_file)  # Expecting list or tensor of [seq_len]
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return input_ids

def measure_io_speed(pt_path, batch_size=32, num_workers_list=[0, 2, 4, 8]):
    results = []

    for num_workers in num_workers_list:
        dataset = LLaMAPTDataset(pt_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        total_bytes = 0
        print(f"\n Measuring I/O speed with num_workers={num_workers}")
        start = time.time()

        for batch in dataloader:
            total_bytes += batch.element_size() * batch.nelement()

        end = time.time()
        duration = end - start
        total_gb = total_bytes / (1024 ** 3)
        speed_gbps = total_gb / duration

        print(f"num_workers={num_workers} --> {speed_gbps:.3f} GB/s over {total_gb:.2f} GB in {duration:.2f} s")
        results.append((num_workers, speed_gbps))

    return results

if __name__ == "__main__":
    #pt_path = "/orcd/datasets/001/shaohao-staging/data-llama/llama_tokenized_data/wikipedia_tokenized.pt"  # Replace with your .pt file
    pt_path = sys.argv[1]  # Replace with your .pt file
    batch_size = 64
    #workers_to_test = [0, 2, 4, 8]
    workers_to_test = [0, 2, 4]

    measure_io_speed(pt_path, batch_size=batch_size, num_workers_list=workers_to_test)

