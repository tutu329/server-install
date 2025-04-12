import os
import torch
import torch.distributed as dist

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    t = torch.tensor([rank], device='cuda')
    dist.all_reduce(t)
    print(f"[Rank {rank}] All-Reduced Value: {t.item()}")

if __name__ == "__main__":
    setup()
