from .Detail_Model_Patch_Custom import Detailer

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Detailer": Detailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Detailer": "Detailer (UNet Patch)",
}
