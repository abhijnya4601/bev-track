"""Fine-tune BEVFormer-tiny's head on the 5-class taxonomy.

    python -m src.train --pretrained ckpts/bevformer_tiny_epoch_24.pth --gpus 1

What this does:
  1. remaps the 10-class checkpoint's classifier rows to 5 classes (src/model.py)
  2. launches BEVFormer's own ``tools/train.py`` under ``torch.distributed.launch``, the same way
     ``tools/dist_train.sh`` does. BEVFormer's train/test loops assume distributed mode even on one GPU.
     It uses ``configs/bevformer_tiny_nusc.py`` with ``load_from`` pointing at the remapped checkpoint
     and in-training validation disabled (the dataset's evaluate() only knows the 10 nuScenes classes).

The training loop itself is BEVFormer's (mmcv EpochBasedRunner). Rewriting it would add risk and
teach nothing about BEV perception. What we control is the config (what is frozen, learning
rates, the class set, the split) and the checkpoint surgery.
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs/bevformer_tiny_nusc.py"))
    ap.add_argument("--pretrained", default=os.path.join(REPO, "ckpts/bevformer_tiny_epoch_24.pth"))
    ap.add_argument("--work-dir", default=os.path.join(REPO, "work_dirs/bevformer_tiny_5cls"))
    ap.add_argument("--bevformer-root", default=os.path.join(REPO, "third_party/BEVFormer"))
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--port", type=int, default=28509)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-remap", action="store_true",
                    help="Skip classifier remapping and let the 5-class head start from random init (ablation).")
    ap.add_argument("--cfg-options", nargs="*", default=[], help="Extra mmcv key=value overrides.")
    ap.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = ap.parse_args(argv)

    if args.no_remap:
        load_from = os.path.abspath(args.pretrained)
    else:
        load_from = os.path.join(REPO, "ckpts/bevformer_tiny_5cls_init.pth")
        if not args.dry_run:
            from src.model import remap_checkpoint

            changed = remap_checkpoint(args.pretrained, load_from)
            print(f"Remapped {len(changed)} classifier tensors -> {load_from}")

    cmd = [sys.executable, "-m", "torch.distributed.launch", f"--nproc_per_node={args.gpus}",
           f"--master_port={args.port}", "tools/train.py", os.path.abspath(args.config),
           "--launcher", "pytorch", "--deterministic", "--no-validate", f"--seed={args.seed}",
           "--work-dir", os.path.abspath(args.work_dir),
           "--cfg-options", f"load_from={load_from}", *args.cfg_options]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([args.bevformer_root, REPO, os.environ.get("PYTHONPATH", "")])}
    print("cd", args.bevformer_root, "&&", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=args.bevformer_root, env=env, check=True)


if __name__ == "__main__":
    main()
