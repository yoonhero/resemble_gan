import torch

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizser, lr):
    print("=> Loading checkpoin")
    checkpoint =torch.load(checkpoint_file, map_location="cuda")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizser.param_groups:
        param_group["lr"] = lr 
