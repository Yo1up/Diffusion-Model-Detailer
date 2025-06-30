import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    """
    A wrapper class that intercepts model calls and allows custom processing
    """
    def __init__(self, original_model, scaling_factor, narrowness):
        super().__init__()
        self.original_model = original_model
        self.narrowness = narrowness
        self.scaling_factor = scaling_factor
        self._is_wrapped = True

    def unwrap(self):
        return self.original_model

    def sin_schedule(self, x):
        new_x = x / 1000 # (x - self.sigma_min) / (self.sigma_max - self.sigma_min)
        return torch.sin(new_x * torch.pi) ** self.narrowness

    def forward(self, *args, **kwargs):
        # PRE-PROCESSING SPACE - Add your custom code here
        # You can modify inputs, log information, etc.
        # Example: print(f"Model called with args: {len(args)}, kwargs: {list(kwargs.keys())}")

        # Call the original model
        result = self.original_model(*args, **kwargs)

        print(args[1])
        timestep = args[1][0]
        print(timestep)
        result = result + (result * self.sin_schedule(timestep) * self.scaling_factor)
        # POST-PROCESSING SPACE - Add your custom code here
        # You can modify outputs, save results, apply transformations, etc.
        # Example: print(f"Model output shape: {result.shape if hasattr(result, 'shape') else type(result)}")

        return result

    def __getattribute__(self, name):
        # Always use direct access for internal stuff
        if name in {"original_model", "scaling_factor", "narrowness", "_is_wrapped", "unwrap"}:
            return object.__getattribute__(self, name)
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # fallback to wrapped model
            return getattr(self.original_model, name)



class DetailModelWrapperNode:
    """
    A ComfyUI node that wraps a model with custom processing capabilities
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "scaling_factor": ("FLOAT", {"default": 0.05, "min": -1.0, "max": 1.0, "step": 0.005}),
                "scaling_narrowness": ("FLOAT", ),
                # "effect_depth": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("wrapped_model",)
    FUNCTION = "wrap_model"
    CATEGORY = "model_processing"

    def unwrap_model(self, wrapped_model):
        """
        Unwraps the input model with custom processing capabilities
        """

        unwrapped_model = wrapped_model.clone()

        # Get the actual model object (usually under 'model' attribute)
        if hasattr(wrapped_model, 'model') and hasattr(wrapped_model.model, 'diffusion_model') and hasattr(wrapped_model.model.diffusion_model, "unwrap"):
            # Wrap the diffusion model (for SD models)
            unwrapped_model.model.diffusion_model = wrapped_model.model.diffusion_model.unwrap()
        elif hasattr(wrapped_model, 'model') and hasattr(wrapped_model.model, "unwrap"):
            # Wrap the main model
            unwrapped_model.model = wrapped_model.model.unwrap()
        elif hasattr(wrapped_model, "unwrap"):
            # Unwrap the whole object
            unwrapped_model = wrapped_model.unwrap()
        else:
            # Fallback to passing model straight through if no unwrapper can be found.
            unwrapped_model = wrapped_model

        return (unwrapped_model,)

    def wrap_model(self, model, scaling_factor, scaling_narrowness):
        """
        Wraps the input model with custom processing capabilities
        """

        model = self.unwrap_model(model.clone())

        # Clone the model to avoid modifying the original
        wrapped_model = model[0].clone()

        # Get the actual model object (usually under 'model' attribute)
        if hasattr(wrapped_model, 'model') and hasattr(wrapped_model.model, 'diffusion_model'):
            # Wrap the diffusion model (for SD models)
            wrapped_model.model.diffusion_model = ModelWrapper(wrapped_model.model.diffusion_model, scaling_factor, scaling_narrowness)
        elif hasattr(wrapped_model, 'model'):
            # Wrap the main model
            wrapped_model.model = ModelWrapper(wrapped_model.model, scaling_factor, scaling_narrowness)
        else:
            # Fallback: wrap the entire model object
            wrapped_model = ModelWrapper(wrapped_model, scaling_factor, scaling_narrowness)

        return (wrapped_model,)

class DetailModelUnwrapperNode:
    """
    A ComfyUI node that wraps a model with custom processing capabilities
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("unwrapped_model",)
    FUNCTION = "unwrap_model"
    CATEGORY = "model_processing"

    def unwrap_model(self, model):
        """
        Unwraps the input model with custom processing capabilities
        """

        unwrapped_model = model.clone()

        # Get the actual model object (usually under 'model' attribute)
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model') and hasattr(model.model.diffusion_model, "unwrap"):
            # Wrap the diffusion model (for SD models)
            unwrapped_model.model.diffusion_model = model.model.diffusion_model.unwrap()
        elif hasattr(model, 'model') and hasattr(model.model, "unwrap"):
            # Wrap the main model
            unwrapped_model.model = model.model.unwrap()
        elif hasattr(model, "unwrap"):
            # Unwrap the whole object
            unwrapped_model = model.unwrap()
        else:
            # Fallback to passing model straight through if no unwrapper can be found.
            unwrapped_model = model

        return (unwrapped_model,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DetailModelWrapperNode": DetailModelWrapperNode,
    "DetailModelUnwrapperNode": DetailModelUnwrapperNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailModelWrapperNode": "Detail Model Wrapper",
    "DetailModelUnwrapperNode": "Detail Model Unwrapper"
}
