"""
Minimal test of torch.autocast's weight cache across optimizer steps.

Several optimizer steps run inside ONE bf16 autocast context, as in the cone training
step under Lightning's "bf16-mixed". If the cached bf16 weight copies go stale after
optimizer.step(), the loss repeats instead of changing.

Expected if the stale-cache explanation is right:
  1. cache on            -> the same loss every step (stale weights)
  2. cache off           -> the loss changes every step
  3. outer cache on,
     inner cache off     -> the loss changes every step (proposed fix)
"""
import torch

assert torch.cuda.is_available(), "run this on a GPU node"
torch.manual_seed(0)
dev = "cuda"

lin = torch.nn.Linear(16, 16).to(dev)
init = {k: v.clone() for k, v in lin.state_dict().items()}
opt = torch.optim.SGD(lin.parameters(), lr=0.5)
x = torch.randn(4, 16, device=dev)


def step():
    opt.zero_grad()
    loss = lin(x).float().pow(2).mean()
    loss.backward()
    opt.step()
    return round(loss.item(), 6)


def run(outer_cache, inner_cache=None, n=4):
    lin.load_state_dict(init)
    losses = []
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=outer_cache):
        for _ in range(n):
            if inner_cache is None:
                losses.append(step())
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=inner_cache):
                    losses.append(step())
    return losses


print("PyTorch", torch.__version__)
print("1. cache on                     :", run(True))
print("2. cache off                    :", run(False))
print("3. outer cache on, inner off    :", run(True, inner_cache=False))
