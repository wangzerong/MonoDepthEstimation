from .midas import model_loader
from .efficient import Efficientnet
import torch
midas_models = {
    "dpt_beit_large_512": "weights/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "weights/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "weights/dpt_beit_base_384.pt",
    "dpt_swin2_large_384": "weights/dpt_swin2_large_384.pt",
    "dpt_swin2_base_384": "weights/dpt_swin2_base_384.pt",
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",
    "dpt_swin_large_384": "weights/dpt_swin_large_384.pt",
    "dpt_next_vit_large_384": "weights/dpt_next_vit_large_384.pt",
    "dpt_levit_224": "weights/dpt_levit_224.pt",
    "dpt_large_384": "weights/dpt_large_384.pt",
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",
    "midas_v21_384": "weights/midas_v21_384.pt",
    "midas_v21_small_256": "weights/midas_v21_small_256.pt",
    "openvino_midas_v21_small_256": "weights/openvino_midas_v21_small_256.xml",
}


def get_model(device = torch.device("cpu"), ckpt_path = None, features = 32, model_type='efficientnet_lite3'):
    if model_type in ["efficientnet_lite3", "efficientnet_lite2"]:
        model = Efficientnet(features=features, backbone_type = model_type)
        return model
    if model_type in midas_models:
        model, midas_transform, net_w, net_h = model_loader.load_model(device, ckpt_path, model_type)
        return model, midas_transform, net_w, net_h
    

if __name__ == "__main__":
    get_model(model_type="dpt_beit_large_512")
