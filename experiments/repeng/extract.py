import dataclasses
import os
import typing
import warnings

import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

from .control import ControlModel, model_layer_list
from .saes import Sae


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]
    undecoded_directions: dict[int, np.ndarray] | None = None

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """
        with torch.inference_mode():
            dirs, _ = read_representations(
                model,
                tokenizer,
                dataset,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, directions=dirs)

    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        sae: Sae,
        dataset: list[DatasetEntry],
        *,
        decode: bool = True,
        method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_center",
        use_residuals: bool = False,
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            sae (saes.Sae): See the `saes` module for how to load this.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                decode (bool, optional): Whether to decode the vector to make it immediately usable.
                    If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                    to use it. Defaults to True.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
                    than ControlVector.train, which defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """

        residual_scales = {}
        def transform_hiddens_with_residuals(hiddens: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
            sae_hiddens = {}
            residuals = {}
            for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
                encoded = sae.layers[k].encode(v)
                decoded = sae.layers[k].decode(encoded)
                sae_hiddens[k] = encoded
                residuals[k] = v - decoded

                residual_scale = 0
                for i in range(len(v)):
                    residual_scale += calculate_residual_scale(decoded[i], residuals[k][i], v[i])
                residual_scale /= len(v)
                residual_scales[k] = residual_scale

            return sae_hiddens, residuals

        def calculate_residual_scale(decoded_direction: np.ndarray, residual: np.ndarray, hiddens: np.ndarray) -> float:
            """
            Calculate appropriate scaling factor for residuals based on relative projections.
            
            Args:
                decoded_direction: The decoded SAE direction
                residual: The normalized residual direction
                hiddens: Original hidden states for this layer [n_samples, hidden_dim]
            """
            # Project original hiddens onto both directions
            dir_proj = np.abs(project_onto_direction(hiddens, decoded_direction))
            res_proj = np.abs(project_onto_direction(hiddens, residual))
            
            # Calculate mean projection magnitudes
            mean_dir_proj = np.mean(dir_proj)
            mean_res_proj = np.mean(res_proj)
            #print(mean_dir_proj, mean_res_proj)
            # Scale residual to match relative magnitude of main direction
            residual_scale = mean_res_proj / mean_dir_proj
            #print(residual_scale)
            return residual_scale

        with torch.inference_mode():
            dirs, residuals = read_representations(
                model,
                tokenizer,
                dataset,
                transform_hiddens=transform_hiddens_with_residuals,
                method=method,
                sae=sae,
                **kwargs,
            )

            final_dirs = {}
            if decode:
                for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
                    final_dirs[k] = sae.layers[k].decode(v)
                    if use_residuals:
                        final_dirs[k] += residuals[k] * residual_scales[k]
            else:
                final_dirs = dirs

        return cls(model_type=model.config.model_type, directions=final_dirs, undecoded_directions=dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

        ```python
        vector = ControlVector.train(...)
        vector.export_gguf("path/to/write/vector.gguf")
        ```
        ```
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except:
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(1 / other)


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap", "mean_center", "sae_topk_center", "sae_topk_diff"] = "pca_diff",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
    k: int = 100,  # Number of top SAE features to keep
    sae: typing.Optional[Sae] = None,  # Added sae parameter
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    raw_residuals = None
    if transform_hiddens is not None:
        layer_hiddens, raw_residuals = transform_hiddens(layer_hiddens)

    # get directions for each layer
    directions: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="extracting directions"):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2

        if method in ["sae_topk_diff", "sae_topk_center"]:
            if sae is None:
                raise ValueError("sae_topk method requires sae parameter")
            if method == "sae_topk_diff":
                # Get difference between positive and negative examples
                diff = h[::2] - h[1::2]
                # Get mean difference
                mean_direction = np.mean(diff, axis=0)
            elif method == "sae_topk_center":
                center = (h[::2] + h[1::2]) / 2
                train = h
                train[::2] -= center
                train[1::2] -= center
                mean_direction = np.mean(train[::2], axis=0)
            
            # Use SAE to get sparse features
            sparse_features = sae.layers[layer].get_topk_features(mean_direction[np.newaxis, :], k)
            sparse_features = sparse_features[0] #remove batch dimension
            sparse_features /= np.linalg.norm(sparse_features) #normalize
            directions[layer] = sparse_features
        else:
            # Original code for other methods
            if method == "pca_diff":
                train = h[::2] - h[1::2]
            elif method == "pca_center" or method == "mean_center":
                center = (h[::2] + h[1::2]) / 2
                train = h
                train[::2] -= center
                train[1::2] -= center
            elif method == "umap":
                train = h
            else:
                raise ValueError("unknown method " + method)

            if method == "mean_center":
                # Take mean of only positive examples (every other starting at index 0)
                # this is the mean direction, as we centered it earlier and every other example is the negative
                mean_direction = np.mean(train[::2], axis=0)
                directions[layer] = mean_direction / np.linalg.norm(mean_direction)
            elif method != "umap":
                np.random.seed(42)
                torch.manual_seed(42)
                
                # Convert to PyTorch tensor and move to GPU
                train_torch = torch.from_numpy(train).cuda()
                
                # Center the data (PyTorch SVD doesn't center automatically)
                train_mean = train_torch.mean(dim=0, keepdim=True)
                train_centered = train_torch - train_mean
                
                # Randomized SVD version
                n_oversamples = 10
                n_iter = 5
                
                # Random projection matrix
                Q = torch.randn(train_torch.shape[1], 1 + n_oversamples, device='cuda')
                
                # Power iteration
                for _ in range(n_iter):
                    Q = train_centered.T @ (train_centered @ Q)
                    Q, _ = torch.linalg.qr(Q)
                
                # Final small SVD
                small_matrix = train_centered @ Q
                U_small, S_small, V_small = torch.svd(small_matrix)
                
                # Extract first principal component
                direction_torch = (Q @ V_small[:, 0]).cpu().numpy()

                # Fix sign: make the maximum magnitude element positive
                max_magnitude_idx = np.argmax(np.abs(direction_torch))
                if direction_torch[max_magnitude_idx] < 0:
                    direction_torch *= -1

                directions[layer] = direction_torch
            else:
                # still experimental so don't want to add this as a real dependency yet
                import umap  # type: ignore

                umap_model = umap.UMAP(n_components=1)
                embedding = umap_model.fit_transform(train).astype(np.float32)
                directions[layer] = np.sum(train * embedding, axis=0) / np.sum(embedding)

            # calculate sign
            projected_hiddens = project_onto_direction(h, directions[layer])

            # order is [positive, negative, positive, negative, ...]
            positive_smaller_mean = np.mean(
                [
                    projected_hiddens[i] < projected_hiddens[i + 1]
                    for i in range(0, len(inputs) * 2, 2)
                ]
            )
            positive_larger_mean = np.mean(
                [
                    projected_hiddens[i] > projected_hiddens[i + 1]
                    for i in range(0, len(inputs) * 2, 2)
                ]
            )

            if positive_smaller_mean > positive_larger_mean:  # type: ignore
                directions[layer] *= -1
        
        if raw_residuals is not None:
            layer_residuals = raw_residuals[layer]
            match (method):
                case "sae_topk_diff" | "pca_diff":
                    # Get difference between positive and negative examples
                    diff = layer_residuals[::2] - layer_residuals[1::2]
                    # Get mean difference
                    mean_residuals = np.mean(diff, axis=0)
                    residuals[layer] = mean_residuals
                case "sae_topk_center" | "pca_center" | "mean_center":
                    center = (layer_residuals[::2] + layer_residuals[1::2]) / 2
                    layer_residuals[::2] -= center
                    mean_residuals = np.mean(layer_residuals[::2], axis=0) #only positive examples
                    residuals[layer] = mean_residuals
                case "umap":
                    raise ValueError("umap method not yet implemented for residuals")
                case _:
                    raise ValueError(f"unknown method when calculating residuals: {method}")

            residuals[layer] /= np.linalg.norm(residuals[layer]) 

    return directions, residuals


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, desc="generating tokens"):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][last_non_padding_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
