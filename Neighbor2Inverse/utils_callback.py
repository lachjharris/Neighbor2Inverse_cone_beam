import lightning as pl
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lightning.pytorch.callbacks import Callback
import yaml
from torchvision.utils import save_image
import pandas as pd

def normalize_to_range(tensor, vmin, vmax):
    """
    Normalize a tensor to the range [vmin, vmax].
    
    Args:
        tensor (torch.Tensor): Input tensor.
        vmin (float): Minimum value for normalization.
        vmax (float): Maximum value for normalization.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    tensor = torch.clamp(tensor, vmin, vmax)  # Clip values outside [vmin, vmax]
    tensor = (tensor - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    return tensor
   


class SavePredictionPE(Callback):
    """
    Callback to save model predictions on a validation dataset slice at the end of each epoch.
    """
    def __init__(self, input_file, output_dir="predictions", example_idx=0):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.example_idx = example_idx
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        input_image = torch.from_numpy(np.load(self.input_file).astype('float32')).to(pl_module.device)
        pl_module.eval()
        device = pl_module.device

        # Get the example input image
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        vmin, vmax = (-600 - 1500/2 + 1024)/4096, (-600 + 1500/2 + 1024)/4096

        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [input_image.squeeze(0).cpu(), predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()

class SavePredictionCallbackSlice(Callback):
    """
    Callback to save model predictions on a validation dataset slice at the end of each epoch.
    """
    def __init__(self, validation_dataset, output_dir="predictions", example_idx=0):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.output_dir = output_dir
        self.example_idx = example_idx
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        pl_module.eval()
        device = pl_module.device

        # Get the example input image
        proj, input_image, pos, exptime = self.validation_dataset.__getitem__(self.example_idx)
        input_image = torch.from_numpy(input_image[:1, 1024:2048, 1024:2048]).to(device).unsqueeze(0)
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        mn, std = input_image.mean().detach().cpu(), input_image.std().mean().detach().cpu()
        vmin=mn-3*std
        vmax=mn+3*std
        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [input_image.squeeze(0).cpu(),  predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()


class SavePredictionCallback(Callback):
    """
    Callback to save model predictions on a specified test image at the end of each epoch.
    """
    def __init__(self, output_dir="predictions", imagepath_inpt="./TestSlices/test_slice_15ms_1800projs.npy", imagepath_target="./TestSlices/test_slice_15ms_900projs.npy", exptime="15ms"):
        super().__init__()
        self.output_dir = output_dir
        self.imagepath_inpt = imagepath_inpt
        self.imagepath_target = imagepath_target
        self.exptime = exptime
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        pl_module.eval()
        device = pl_module.device

        # Check if files exist before loading
        if not os.path.exists(self.imagepath_target):
            print(f"Warning: Target image file not found: {self.imagepath_target}")
        
        if not os.path.exists(self.imagepath_inpt):
            print(f"Warning: Input image file not found: {self.imagepath_inpt}")

        # Get the example input image
        try:
            fullview_image = np.load(self.imagepath_target).astype('float32')
            fullview_image = torch.from_numpy(fullview_image[1524:2548, 3024:4048]).unsqueeze(0).unsqueeze(0)
            input_image = np.load(self.imagepath_inpt).astype('float32')
            input_image = torch.from_numpy(input_image[1524:2548, 3024:4048]).to(device).unsqueeze(0).unsqueeze(0)
        except Exception as e:
            print(f"Warning: Error loading image files: {e}")
            return

        fullview_image = pl_module.normalize(fullview_image, pos=[3], exptime=[str(self.exptime)])
        input_image = pl_module.normalize(input_image, pos=[3], exptime=[str(self.exptime)])
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        mn, std = input_image.mean().detach().cpu(), input_image.std().mean().detach().cpu()
        vmin=mn-3*std
        vmax=mn+3*std
        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [fullview_image.squeeze(0), input_image.squeeze(0).cpu(),  predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()

def save_cone_progress_panels(path_png, path_tif, inpt, pred, fov, residual_range, title,
                              scan_mean, scan_std, show_residual=True, save_tiff=True):
    """
    Draw [input, prediction, (residual)] for SavePredictionCallbackCone (numpy only, no torch).
    All arrays are 2-D float, physical units (same units as the full-res reconstruction).

    PNG: input and prediction share the input's window (FOV mean +- 3 std, as before); the residual
    (input - prediction) uses the FIXED symmetric window +- residual_range, so its brightness is
    comparable between epochs and runs. Each panel has a colourbar; the residual title gives its
    std (also as a fraction of the scan std used for normalisation) and the fraction of FOV pixels
    clipped by the window.
    TIFF (save_tiff): float32 ImageJ stack of the same panels, unscaled, for Fiji.
    Returns a dict of the residual numbers (None if show_residual is False).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import tifffile

    vals = inpt[fov]
    vmin, vmax = float(vals.mean() - 3 * vals.std()), float(vals.mean() + 3 * vals.std())
    panels = [(inpt, "input", vmin, vmax), (pred, "prediction", vmin, vmax)]
    stats = None
    if show_residual:
        res = inpt - pred
        r = float(residual_range)
        res_fov = res[fov]
        stats = {
            "std": float(res_fov.std()),
            "mean": float(res_fov.mean()),
            "std_over_scan_std": float(res_fov.std() / scan_std),
            "clipped_fraction": float((np.abs(res_fov) > r).mean()),
        }
        panels.append((res, (f"residual (input - prediction)\n"
                             f"std {stats['std']:.4g} = {stats['std_over_scan_std']:.4f} scan std, "
                             f"mean {stats['mean']:.3g}\n"
                             f"fixed window +-{r:.4g}, clipped {100 * stats['clipped_fraction']:.2f}% of FOV"),
                       -r, r))

    fig = Figure(figsize=(7.2 * len(panels), 7.8), dpi=100)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(panels))
    for ax, (img, label, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("reconstruction units")
    fig.suptitle(f"{title}   (input/prediction window {vmin:.4g} .. {vmax:.4g}; "
                 f"scan mean {scan_mean:.4g}, std {scan_std:.4g})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path_png)

    if save_tiff:
        labels = ["input", "prediction"] + (["residual (input - prediction)"] if show_residual else [])
        tifffile.imwrite(path_tif, np.stack([p[0] for p in panels]).astype(np.float32),
                         imagej=True, metadata={"Labels": labels})
    return stats


class SavePredictionCallbackCone(Callback):
    """
    Cone-beam counterpart of SavePredictionCallback: at train start and after every
    validation epoch, denoise one slice of a saved FULL-RESOLUTION reconstruction
    (from 0_calculateStatsCone.py --save_dir) and save [input, prediction, residual].

    There is no high-dose reference for the cone data, so the residual
    (input - prediction) takes the place of upstream's reference panel: it should
    look like structureless noise; visible anatomy means structure is being removed.

    Outputs per call (physical units, see save_cone_progress_panels):
      <name>_<epoch>_<scan>_slice<k>_progress.png  panels with colourbars; the residual has a FIXED
                                                   window, so its brightness compares across epochs/runs
      <name>_<epoch>_<scan>_slice<k>_progress.tif  float32 stack for Fiji (save_tiff: True)
    residual_range: half-width of the residual window (reconstruction units). null -> 3 x a rough
    noise estimate of the input slice (horizontal neighbour differences in the FOV), fixed on the
    first call. It depends only on the input slice, so runs on the same slice get the same window.
    The residual numbers are also printed to the .out ("prediction callback ...").
    """
    def __init__(self, output_dir, volume_path, scan_id, slice_idx=None, show_residual=True,
                 residual_range=None, save_tiff=True):
        super().__init__()
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Full-res reconstruction not found: {volume_path}")

        volume = np.load(volume_path, mmap_mode='r')  # [slice, y, x]; only one slice is read
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2  # central slice: least cone-beam artefact
        self.slice = np.array(volume[slice_idx], dtype=np.float32)
        del volume

        # Display window is taken from inside the circular FOV only,
        # so the zeroed corners don't bias mean/std.
        self.fov = self.slice != 0

        self.output_dir = output_dir
        self.scan_id = scan_id
        self.slice_idx = slice_idx
        self.show_residual = show_residual
        self.save_tiff = save_tiff
        if residual_range is None:
            self.residual_range = 3 * self.noise_estimate(self.slice, self.fov)
            source = "3 x neighbour-difference noise estimate of the input slice"
        else:
            self.residual_range = float(residual_range)
            source = "from the YAML"
        print(f"prediction callback: residual window +-{self.residual_range:.5g} ({source})")
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def noise_estimate(img, fov):
        """
        Rough noise std of img from horizontal neighbour differences inside the FOV: std(d) / sqrt(2).
        Neighbouring voxels of an FDK reconstruction are correlated and edges add to d, so this is
        not an exact noise level; it only fixes a display range that is the same for every epoch.
        """
        both = fov[:, 1:] & fov[:, :-1]
        d = (img[:, 1:] - img[:, :-1])[both]
        return float(d.std() / np.sqrt(2))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.save_current_progress(trainer, pl_module, "epoch")

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.save_current_progress(trainer, pl_module, "start")

    def save_current_progress(self, trainer, pl_module, name):
        from modelLightning import pad_to_divisible, unpad_from_divisible

        was_training = pl_module.training
        pl_module.eval()

        input_image = torch.from_numpy(self.slice).to(pl_module.device)[None, None]  # [1, 1, H, W]
        input_image = pl_module.normalize_cone(input_image, self.scan_id)

        with torch.no_grad():
            inpt_pad, pad = pad_to_divisible(input_image, 32)
            predicted_image = unpad_from_divisible(pl_module(inpt_pad), pad)

        # Back to physical units (the scan stats used by normalize_cone).
        mean, std = pl_module.cone_scan_stats(self.scan_id)
        pred = predicted_image[0, 0].float().cpu().numpy() * std + mean
        del input_image, inpt_pad, predicted_image

        stem = os.path.join(self.output_dir,
                            f"{name}_{trainer.current_epoch}_{self.scan_id}_slice{self.slice_idx}_progress")
        stats = save_cone_progress_panels(
            stem + ".png", stem + ".tif", self.slice, pred, self.fov, self.residual_range,
            title=f"{name} {trainer.current_epoch}: {self.scan_id} slice {self.slice_idx}",
            scan_mean=mean, scan_std=std, show_residual=self.show_residual, save_tiff=self.save_tiff,
        )
        if stats is not None:
            print(f"prediction callback {name} {trainer.current_epoch}: residual std {stats['std']:.5g} "
                  f"({stats['std_over_scan_std']:.4f} scan std), mean {stats['mean']:.4g}, "
                  f"clipped {100 * stats['clipped_fraction']:.2f}% of FOV")

        if was_training:
            pl_module.train()


class SaveHyperparametersCallback(Callback):
    def __init__(self, output_dir, file):
        super().__init__()
        self.output_dir = output_dir
        self.file = file

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            # Save hyperparameter file to the weights directory
            with open(f'{self.output_dir}/trainparams.yml', 'w') as outfile:
                yaml.dump(self.file, outfile, default_flow_style=False)


class SavePredictionCallbackProj(Callback):
    """
    Callback to save model predictions on a validation dataset proj at the end of each epoch.
    """
    def __init__(self, validation_dataset, output_dir="predictions", example_idx=0):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.output_dir = output_dir
        self.example_idx = example_idx
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        pl_module.eval()
        device = pl_module.device

        # Get the example input image
        input_image = self.validation_dataset.__getitem__(self.example_idx)
        input_image = torch.from_numpy(input_image[:1]).to(device).unsqueeze(0)
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        mn, std = input_image.mean().detach().cpu(), input_image.std().mean().detach().cpu()
        vmin=mn-3*std
        vmax=mn+3*std
        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [input_image.squeeze(0).cpu(),  predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()
