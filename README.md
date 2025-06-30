# Diffusion-Model-Detailer
ComfyUI nodes that allow the user to control the generation of diffusion models to increase and decrease level of detail. The model patch has no trainable parameters and can be applied to theoretically any diffusion model in existence. whether or not the implementation currently works for every diffusion model in existence is unknown.

# Installation
put the python script directly in the custom_nodes folder. if you place it in a folder it won't load properly.

# Planned Features
- [ ] Add a "depth" parameter to alter when the scheduled noise injection/removal peaks.

- [ ] Migrate to using the proper model patcher comfy commands.

- [ ] Make the nodes able to run inside a folder so comfy manager will be able to install the nodes automatically.

- [ ] Add some example image grids

- [ ] Add some example workflows

