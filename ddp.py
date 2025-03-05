import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size, master_addr):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "12355"  # Any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, master_addr):
    setup(rank, world_size, master_addr)

    model = nn.Linear(10, 10).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters())

    for _ in range(100):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).cuda(rank))
        labels = torch.randn(20, 10).cuda(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--master_addr", type=str)
    args = parser.parse_args()
    main(args.rank, args.world_size, args.master_addr)