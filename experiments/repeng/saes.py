import dataclasses
import json
import pathlib
import typing

import numpy as np
import torch
import torch.types
import tqdm


class SaeLayer(typing.Protocol):
    def encode(self, activation: np.ndarray) -> np.ndarray: ...
    def decode(self, features: np.ndarray) -> np.ndarray: ...


@dataclasses.dataclass
class Sae:
    layers: dict[int, SaeLayer]

    def steer_with_features(self, feature_dict: dict[int, dict[int, float]], model_type: str = "llama") -> "ControlVector":
        """Create a steering vector by directly activating SAE features.
        
        Args:
            feature_dict: Dictionary mapping layer IDs to {feature_id: strength} dicts
            model_type: The model type for the ControlVector (default: "llama")
            
        Returns:
            ControlVector with the specified features activated
            
        Example:
            # Activate feature 123 in layer 24 with strength 0.5
            # and feature 456 in layer 30 with strength -0.3
            vector = sae.steer_with_features({
                24: {123: 0.5},
                30: {456: -0.3}
            })
        """
        from repeng.extract import ControlVector
        
        steering_vectors = {}
        
        for layer_id, features in feature_dict.items():
            if layer_id not in self.layers:
                continue
                
            # Create zero vector in feature space
            feature_vec = np.zeros(self.layers[layer_id].sae.num_latents, dtype=np.float32)
            
            # Set specified feature strengths
            for feature_id, strength in features.items():
                feature_vec[feature_id] = strength
                
            # Decode back to activation space
            steering_vectors[layer_id] = self.layers[layer_id].decode(feature_vec)
            
        # Create and return ControlVector
        return ControlVector(
            model_type=model_type,
            directions=steering_vectors,
            undecoded_directions={k: v for k, v in feature_dict.items() if k in steering_vectors}
        )

def from_eleuther(
    device: str = "cpu",  # saes wants str | torch.device, safetensors wants str | int... so str it is
    dtype: torch.dtype | None = torch.bfloat16,
    layers: typing.Iterable[int] = range(1, 32),
    repo: str = "EleutherAI/sae-llama-3-8b-32x",
    revision: str = "32926540825db694b6228df703f4528df4793d67",
) -> Sae:
    """
    Note that `layers` should be 1-indexed, repeng style, not 0-indexed, Eleuther style. This may change in the future.

    (Context: repeng counts embed_tokens as layer 0, then the first transformer block as layer 1, etc. Eleuther
    counts embed_tokens separately, then the first transformer block as layer 0.)
    """

    try:
        import safetensors.torch, huggingface_hub
        import sae as eleuther_sae  # type: ignore
    except ImportError as e:
        raise ImportError(
            "`sae` (or a transitive dependency) not installed"
            "--please install `sae` and its dependencies from https://github.com/EleutherAI/sae"
        ) from e

    @dataclasses.dataclass
    class EleutherSaeLayer:
        # see docstr
        # hang on to both for debugging
        repeng_layer: int
        eleuther_layer: int
        sae: eleuther_sae.Sae

        def encode(self, activation: np.ndarray) -> np.ndarray:
            # TODO: this materializes the entire, massive feature vector in memory
            # ideally, we would sparsify like the sae library does--need to figure out how to run PCA on the sparse matrix
            at = torch.from_numpy(activation).to(self.sae.device)
            out = self.sae.pre_acts(at)
            # numpy doesn't like bfloat16
            return out.cpu().float().numpy()

        def get_topk_features(self, features: np.ndarray, k: int) -> np.ndarray:
            """Get a sparse vector containing only the top k features from SAE activations.
            
            Args:
                activation: Input activation tensor
                k: Number of top features to keep
                
            Returns:
                Sparse vector with only top k features preserved
            """
            # Get indices of top k features by absolute value
            top_k_indices = np.abs(features).argsort(axis=-1)[:, -k:]
            
            # Create mask of zeros with ones at top k indices
            mask = np.zeros_like(features)
            # Handle batched input
            for i in range(features.shape[0]):
                mask[i, top_k_indices[i]] = 1
                
            # Apply mask to keep only top k features
            sparse_features = features * mask
            
            return sparse_features

        def decode(self, features: np.ndarray) -> np.ndarray:
            # TODO: see encode, this is not great. `sae` ships with kernels for doing this sparsely, we should use them
            ft = torch.from_numpy(features).to(self.sae.device, dtype=dtype)
            decoded = ft @ self.sae.W_dec
            return decoded.cpu().detach().float().numpy()

    # TODO: only download requested layers?
    base_path = pathlib.Path(huggingface_hub.snapshot_download(repo, revision=revision))
    layer_dict: dict[int, SaeLayer] = {}
    for layer in tqdm.tqdm(layers):
        eleuther_layer = layer - 1 # see docstr
        # this is in `sae` but to load the dtype we want, need to reimpl some stuff
        layer_path = base_path / f"layers.{eleuther_layer}"
        with (layer_path / "cfg.json").open() as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            if 'signed' in cfg_dict:
                del cfg_dict['signed']
            cfg = eleuther_sae.SaeConfig(**cfg_dict)

        layer_sae = eleuther_sae.Sae(d_in, cfg, device=device, dtype=dtype)
        safetensors.torch.load_model(
            model=layer_sae,
            filename=layer_path / "sae.safetensors",
            device=device,
            strict=True,
        )
        # repeng counts embed_tokens as layer 0 and further layers as 1, 2, ...
        # eleuther counts embed_tokens separately and further layers as 0, 1, ...
        layer_dict[layer] = EleutherSaeLayer(
            repeng_layer=layer, eleuther_layer=eleuther_layer, sae=layer_sae
        )

    return Sae(layers=layer_dict)
