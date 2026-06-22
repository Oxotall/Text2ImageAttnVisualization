"""attnviz — visualize attention and cross-attention in Stable Diffusion 1.5.

The package decomposes a text-to-image model into its three components
(text encoder, VAE, UNet) and captures the attention maps the UNet computes
while denoising, so they can be visualized:

* cross-attention: how much each image region attends to each text token,
* self-attention: how much each image region attends to other image regions.
"""

from .config import Config
from .device_resolver import DeviceResolver
from .pipeline_loader import PipelineLoader
from .attention_store import AttentionStore
from .attention_controller import AttentionController
from .token_decoder import TokenDecoder
from .generator import Generator, GenerationResult
from .cross_attention_visualizer import CrossAttentionVisualizer
from .self_attention_visualizer import SelfAttentionVisualizer
from .image_to_text_visualizer import ImageToTextVisualizer, TokenWeight

__all__ = [
    "Config",
    "DeviceResolver",
    "PipelineLoader",
    "AttentionStore",
    "AttentionController",
    "TokenDecoder",
    "Generator",
    "GenerationResult",
    "CrossAttentionVisualizer",
    "SelfAttentionVisualizer",
    "ImageToTextVisualizer",
    "TokenWeight",
]
