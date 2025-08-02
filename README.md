# Diffusion-Model-Detailer
ComfyUI nodes that allow the user to control the generation of diffusion models to increase and decrease level of detail. The model patch has no trainable parameters and can be applied to theoretically any diffusion model in existence. whether or not the implementation currently works for every diffusion model in existence is unknown.

# Installation
You can now install it with the ComfyUI manager. if you experience any issues please let me know.

# Usage
## Detail model wrapper node
 - **scaling factor**: determines the strength and "direction" of the modification. positive values increase the denoising model's agressiveness during affected steps and negative decreases it's aggressiveness at affected steps. this in turn means that positive values decrease "detail" and negative values increase "detail"
 - **scaling narrowness**: this determines how "narrow" (or wide) the affected steps are. a value of 0 affects every single step at the strength of the scaling factor. higher values decrease the width (or increases the "narrowness") of the affected step area according to an exponentiated sine wave (i.e. the schedule is a sine wave raised to the power of "scaling narrowness")
## Detail model unwrapper node
 - Unwraps the model, this must be used **_every time you want to remove the applied model even if you don't have the wrapper in the node tree, if it was loaded last it will remain loaded until it is unwrapped or re-wrapped_**
 - once I figure out how to get it to work properly in comfyUIs proper wrapper framework I will remove this node as it will be unnecessary.

# Planned Features
- [ ] Add a "depth" parameter to alter when the scheduled noise injection/removal peaks.

- [ ] Migrate to using the proper model patcher comfy commands.

- [*] Make the nodes able to run inside a folder so comfy manager will be able to install the nodes automatically.

- [ ] Add some example image grids

- [ ] Add some example workflows

