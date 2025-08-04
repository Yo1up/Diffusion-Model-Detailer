import torch
import torch.nn.functional as F
import comfy.model_patcher
import comfy.samplers
import math

class Detailer:
    """
    A ComfyUI node that uses the `set_model_unet_function_wrapper` hook to precisely modify
    the raw noise prediction output from the UNet model. This allows for fine-grained
    control over detail before or during the CFG calculation.

    - A positive `scaling_factor` amplifies details by increasing the residual noise in the output.
    - A negative `scaling_factor` smoothens details by dampening the residual noise in the output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scaling_factor": ("FLOAT", {
                    "default": 0.25,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.005,
                    "display": "slider",
                    "tooltip": "Positive values increase detail, negative values reduce detail."
                }),
                "scaling_narrowness": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 14.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Controls the focus of the scaling effect within the sampling process (higher = narrower peak)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/details" # This defines where it appears in the ComfyUI menu

    def patch(self, model, scaling_factor, scaling_narrowness):
        """
        Applies a patch to the model's UNet function to modify noise predictions
        based on a sine wave schedule and a scaling factor.
        """

        # we scale the user input to make it an acceptable sigma modification so the user doesn't have to work with very small numbers'
        actual_scaling_factor = scaling_factor * 0.05

        def sin_schedule_for_unet(timestep_tensor, current_sigmas):
            """
            Calculates a scaling multiplier based on the current timestep (noise level).
            This function creates a sine wave curve across the sampling steps,
            allowing the effect to be strongest in the middle of the process.
            """
            # ComfyUI's `model.apply_model`
            # typically passes the current 'sigma' value as the 't' (timestep_tensor) argument.
            boolean_mask = (timestep_tensor.item() == current_sigmas)
            steps = len(current_sigmas) - 1
            step = torch.argwhere(boolean_mask)
            current_sigmas = timestep_tensor

            normalized_progress = step / steps

            # Clamp the normalized sigma to ensure it stays within [0, 1] just in case my code is too jank
            normalized_progress_clamped = torch.clamp(normalized_progress, 0.0, 1.0)

            # Apply the sine wave schedule. The `scaling_narrowness`
            # controls the sharpness of the peak of the sine wave.
            return torch.sin(normalized_progress_clamped * math.pi) ** scaling_narrowness

        def apply_detailer(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            cond_or_uncond = args["cond_or_uncond"]
            c = args["c"]


            mod_strength = sin_schedule_for_unet(timestep[0], c["transformer_options"]["sample_sigmas"])
            timestep[0] = torch.mul(timestep[0], (1 + (mod_strength * actual_scaling_factor)))

            noise_pred = apply_model(input_x, timestep, **c)

            return noise_pred

        # Clone the model to ensure that this patch only affects the current branch
        # of the ComfyUI workflow and does not modify the original model in place. idk if this is really necessary but I don't think it can hurt.
        model_clone = model.clone()

        # Apply the custom `apply_detailer` method using ComfyUI's patching mechanism.
        # This hooks into the model's internal UNet forward pass.
        model_clone.set_model_unet_function_wrapper(apply_detailer)

        return (model_clone,)

# Node registration for ComfyUI.
# These dictionaries tell ComfyUI how to display and interact with your custom node.
NODE_CLASS_MAPPINGS = {
    "Detailer": Detailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Detailer": "Detailer (UNet Patch)",
}
