import DetailModelWrapperNode from Detail_Model_Patch_Custom
import DetailModelUnwrapperNode from Detail_Model_Patch_Custom

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DetailModelWrapperNode": DetailModelWrapperNode,
    "DetailModelUnwrapperNode": DetailModelUnwrapperNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailModelWrapperNode": "Detail Model Wrapper",
    "DetailModelUnwrapperNode": "Detail Model Unwrapper"
}
