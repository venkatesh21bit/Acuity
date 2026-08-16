import json
import os

# --- 1. Fix train.py ---
with open("train.py", "r", encoding="utf-8") as f:
    train_src = f.read()

train_src = train_src.replace(
    "torch.cuda.amp.autocast(dtype=amp_dtype, enabled=(amp_dtype != torch.float32))",
    "torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32))"
)
train_src = train_src.replace(
    "agg.update(pred.cpu(), gt_imgs.cpu())",
    "agg.update(pred.cpu().float(), gt_imgs.cpu().float())"
)
train_src = train_src.replace(
    "def train_one_epoch(\n    model:     nn.Module,\n    criterion: CompoundLoss,\n    loader,\n    optimizer,\n    scheduler,\n    log_vst:   LogVST,\n    device:    torch.device,\n    epoch:     int,\n    cfg:       dict,\n    amp_dtype: torch.dtype,\n) -> dict:",
    "def train_one_epoch(\n    model:     nn.Module,\n    criterion: CompoundLoss,\n    loader,\n    optimizer,\n    scheduler,\n    scaler:    torch.cuda.amp.GradScaler,\n    log_vst:   LogVST,\n    device:    torch.device,\n    epoch:     int,\n    cfg:       dict,\n    amp_dtype: torch.dtype,\n) -> dict:"
)
train_src = train_src.replace(
    "best_psnr  = -float(\"inf\")",
    "scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))\n    best_psnr  = -float(\"inf\")"
)
train_src = train_src.replace(
    "train_stats = train_one_epoch(\n            model, criterion, train_loader, optimizer, scheduler,\n            log_vst, device, epoch, cfg, amp_dtype,\n        )",
    "train_stats = train_one_epoch(\n            model, criterion, train_loader, optimizer, scheduler, scaler,\n            log_vst, device, epoch, cfg, amp_dtype,\n        )"
)
old_backward = """        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [criterion.log_vars],
            max_norm=grad_clip,
        )
        optimizer.step()"""
new_backward = """        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [criterion.log_vars],
            max_norm=grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()"""
train_src = train_src.replace(old_backward, new_backward)

with open("train.py", "w", encoding="utf-8") as f:
    f.write(train_src)


# --- 2. Fix infer.py ---
with open("infer.py", "r", encoding="utf-8") as f:
    infer_src = f.read()

infer_src = infer_src.replace("torch.cuda.amp.autocast", "torch.amp.autocast")
infer_src = infer_src.replace("torch.amp.autocast(dtype=", "torch.amp.autocast('cuda', dtype=")

with open("infer.py", "w", encoding="utf-8") as f:
    f.write(infer_src)


# --- 3. Fix Notebook ---
nb_path = "NAFNet_SpeckleRestoration.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        
        src = src.replace("torch.cuda.amp.autocast", "torch.amp.autocast")
        src = src.replace("torch.amp.autocast(dtype=", "torch.amp.autocast('cuda', dtype=")
        
        if "cell-train-inline" in cell["id"]:
            src = src.replace("agg.update(pred.cpu(), gt_imgs.cpu())", "agg.update(pred.cpu().float(), gt_imgs.cpu().float())")
            
            if "scaler = " not in src:
                src = src.replace("global_step = 0", "global_step = 0\nscaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))")
            
            old_nb_backward = """        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [criterion.log_vars],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()"""
            
            new_nb_backward = """        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [criterion.log_vars],
            max_norm=GRAD_CLIP,
        )
        scaler.step(optimizer)
        scaler.update()"""
            src = src.replace(old_nb_backward, new_nb_backward)
            
        if "\n" in src:
            lines = src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        else:
            cell["source"] = [src]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("All fixes applied!")
