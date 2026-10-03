import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
import torch
from torch import Tensor, nn

from .comps.aggregation import ExpertAggregation
from .comps.heads import AdditiveExpertRegressionHead, MultiTaskRegressionHead, TaskGatedRegressionHead
from .comps.moe import MetadataMixtureOfExperts
from .comps.tokenizer import IDPTokenizer

DEFAULT_TARGET_NAMES = ("participant.p21003_i2",)


def model_predict(idps: np.ndarray, pre_trained: str | Path | None = None) -> float:
    """Predict one subject's brain age in years from 3,630 raw-unit IDPs.

    Input order is assets/idp_metadata.csv row order. Without pre_trained,
    model parameters are randomly initialized (demonstration only).
    """
    if not isinstance(idps, np.ndarray) or idps.shape != (3630,) or idps.dtype.kind not in "fiu":
        raise ValueError("idps must be a numeric NumPy array with shape (3630,).")
    values = idps.astype(np.float64)
    missing = np.isnan(values)
    if np.isinf(values).any() or missing.mean() > 0.2:
        raise ValueError("IDPs must have no infinities and at most 20% missing values (NaN).")

    root = Path(__file__).resolve().parent
    config = json.loads((root / "assets/architecture.json").read_text())
    with np.load(root / "assets/preprocessing.npz") as stats:
        preprocessing = dict(stats)
    seed = 2026
    weights = None
    if pre_trained is not None:
        checkpoint = torch.load(pre_trained, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format.")
        config, seed = checkpoint["architecture"], checkpoint["seed"]
        preprocessing = {key: value.numpy() for key, value in checkpoint["preprocessing"].items()}
        weights = checkpoint["state_dict"]

    network = build_recap_model(
        root / "assets/idp_metadata.csv",
        root / "assets/metadata_embeddings.pt",
        root / f"assets/routing_seed{seed}.pt",
        architecture=RecapArchitectureConfig(**config),
    )
    if weights is not None:
        if tuple(checkpoint["feature_names"]) != network.idp_short_names:
            raise ValueError("Checkpoint feature order does not match metadata.")
        network.load_state_dict(weights, strict=True)
    network.eval()
    p = preprocessing
    values = (np.where(missing, p["feature_medians"], values) - p["feature_means"]) / p["feature_stds"]
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Standardized IDPs exceed float32 range.")
    with torch.inference_mode():
        prediction = network(torch.from_numpy(values[None]), torch.from_numpy(missing[None])).predictions.item()
    return float(prediction * np.asarray(p["age_std"]).item() + np.asarray(p["age_mean"]).item())


@dataclass(frozen=True)
class RecapArchitectureConfig:
    """Serializable hyperparameters used to construct a complete RECAP model."""

    model_dim: int = 128
    router_dim: int = 128
    metadata_hidden_dim: int = 256
    value_hidden_dim: int = 64
    tokenizer_dropout: float = 0.1
    use_idp_identity_embedding: bool = True
    expert_hidden_dim: int = 256
    num_routed_experts: int = 12
    top_k: int = 2
    router_hidden_dim: int | None = None
    router_temperature: float = 1.0
    router_anchor_label_smoothing: float = 0.2
    expert_dropout: float = 0.1
    router_dropout: float = 0.0
    moe_output_dropout: float = 0.1
    aggregation_num_layers: int = 2
    aggregation_num_heads: int = 4
    aggregation_feedforward_dim: int | None = None
    aggregation_dropout: float = 0.1
    head_hidden_dim: int | None = None
    head_dropout: float = 0.1
    use_additive_head: bool = False
    use_task_gated_head: bool = True
    shared_only_global: bool = True


def compute_metadata_fingerprint(short_names: list[str], descriptions: list[str]) -> str:
    """Computes the ordered metadata fingerprint used by the embedding script."""
    records = [
        {"IDP short name": short_name, "IDP description": description}
        for short_name, description in zip(short_names, descriptions, strict=True)
    ]
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_recap_model(
    idp_metadata_path: str | Path,
    metadata_embeddings_path: str | Path,
    routing_initialization_path: str | Path,
    task_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    architecture: RecapArchitectureConfig | None = None,
) -> "RecapModel":
    """Loads static artifacts, validates alignment, and constructs RECAP."""
    architecture = RecapArchitectureConfig() if architecture is None else architecture
    metadata = pd.read_csv(idp_metadata_path)
    required_columns = {"IDP short name", "IDP description", "is_routable"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"IDP metadata is missing columns: {sorted(missing_columns)}")

    short_names = metadata["IDP short name"].astype("string").str.strip().tolist()
    descriptions = metadata["IDP description"].astype("string").str.strip().tolist()
    if any(pd.isna(name) or not name for name in short_names):
        raise ValueError("IDP metadata contains empty short names.")
    if any(pd.isna(description) or not description for description in descriptions):
        raise ValueError("IDP metadata contains empty descriptions.")
    if len(set(short_names)) != len(short_names):
        raise ValueError("IDP metadata contains duplicated short names.")

    checkpoint = torch.load(metadata_embeddings_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Metadata embedding checkpoint must contain a dictionary.")
    embeddings = checkpoint.get("metadata_embeddings")
    checkpoint_names = checkpoint.get("idp_short_names")
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise ValueError("Checkpoint metadata_embeddings must be a two-dimensional tensor.")
    if checkpoint_names != short_names:
        raise ValueError("Embedding checkpoint IDP names or order do not match idp_metadata.csv.")
    expected_fingerprint = compute_metadata_fingerprint(short_names, descriptions)
    if checkpoint.get("metadata_sha256") != expected_fingerprint:
        raise ValueError("Embedding checkpoint metadata fingerprint does not match idp_metadata.csv.")
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embedding row count does not match IDP metadata.")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Metadata embeddings contain non-finite values.")
    embedding_norms = embeddings.float().norm(dim=1)
    if not torch.allclose(embedding_norms, torch.ones_like(embedding_norms), atol=1e-4, rtol=1e-4):
        raise ValueError("Metadata embeddings must be row-wise L2-normalized.")

    routed_values = metadata["is_routable"]
    if routed_values.isna().any() or not routed_values.isin([True, False]).all():
        raise ValueError("is_routable must contain only Boolean values.")
    routed_mask = torch.tensor(routed_values.to_numpy(dtype=bool), dtype=torch.bool)

    routing_artifact = torch.load(routing_initialization_path, map_location="cpu", weights_only=True)
    if not isinstance(routing_artifact, dict):
        raise ValueError("Routing initialization artifact must contain a dictionary.")
    if routing_artifact.get("idp_short_names") != short_names:
        raise ValueError("Routing initialization IDP names or order do not match metadata.")
    if routing_artifact.get("metadata_sha256") != expected_fingerprint:
        raise ValueError("Routing initialization metadata fingerprint does not match metadata.")
    initial_assignments = routing_artifact.get("initial_assignments")
    expert_names = tuple(routing_artifact.get("expert_names", ()))
    if routing_artifact.get("num_clusters") != architecture.num_routed_experts:
        raise ValueError("Routing initialization cluster count does not match num_routed_experts.")
    if not isinstance(initial_assignments, Tensor) or tuple(initial_assignments.shape) != (len(metadata),):
        raise ValueError("Routing initial_assignments must have shape [num_idps].")
    if len(expert_names) != architecture.num_routed_experts or len(set(expert_names)) != len(expert_names):
        raise ValueError("Routing expert_names do not match num_routed_experts.")
    initial_assignments = initial_assignments.to(dtype=torch.long)
    if torch.any(routed_mask & (initial_assignments < 0)) or torch.any(~routed_mask & (initial_assignments != -1)):
        raise ValueError("Routing assignments and is_routable disagree.")
    if torch.any(initial_assignments >= architecture.num_routed_experts):
        raise ValueError("Routing initialization contains an out-of-range expert index.")
    expert_eligibility_mask = routed_mask[:, None].expand(-1, architecture.num_routed_experts).clone()

    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("task_names must be non-empty and unique.")

    tokenizer = IDPTokenizer(
        metadata_embeddings=embeddings,
        model_dim=architecture.model_dim,
        router_dim=architecture.router_dim,
        metadata_hidden_dim=architecture.metadata_hidden_dim,
        value_hidden_dim=architecture.value_hidden_dim,
        dropout=architecture.tokenizer_dropout,
        use_idp_identity_embedding=architecture.use_idp_identity_embedding,
    )
    mixture_of_experts = MetadataMixtureOfExperts(
        model_dim=architecture.model_dim,
        router_metadata_dim=architecture.router_dim,
        expert_hidden_dim=architecture.expert_hidden_dim,
        num_routed_experts=architecture.num_routed_experts,
        top_k=architecture.top_k,
        router_hidden_dim=architecture.router_hidden_dim,
        router_temperature=architecture.router_temperature,
        anchor_label_smoothing=architecture.router_anchor_label_smoothing,
        expert_dropout=architecture.expert_dropout,
        router_dropout=architecture.router_dropout,
        output_dropout=architecture.moe_output_dropout,
        shared_only_global=architecture.shared_only_global,
        initial_assignments=initial_assignments,
    )
    aggregation = ExpertAggregation(
        model_dim=architecture.model_dim,
        num_experts=architecture.num_routed_experts,
        num_layers=architecture.aggregation_num_layers,
        num_heads=architecture.aggregation_num_heads,
        feedforward_dim=architecture.aggregation_feedforward_dim,
        dropout=architecture.aggregation_dropout,
        shared_only_global=architecture.shared_only_global,
    )
    if architecture.use_additive_head and architecture.use_task_gated_head:
        raise ValueError("use_additive_head and use_task_gated_head cannot both be true.")
    if architecture.use_additive_head:
        prediction_head: MultiTaskRegressionHead | TaskGatedRegressionHead | AdditiveExpertRegressionHead = (
            AdditiveExpertRegressionHead(
                input_dim=architecture.model_dim,
                num_expert_tokens=architecture.num_routed_experts + 1,
                task_names=task_names,
            )
        )
    elif architecture.use_task_gated_head:
        prediction_head = TaskGatedRegressionHead(
            input_dim=architecture.model_dim,
            num_expert_tokens=architecture.num_routed_experts + 1,
            task_names=task_names,
        )
    else:
        prediction_head = MultiTaskRegressionHead(
            input_dim=architecture.model_dim,
            task_names=task_names,
            hidden_dim=architecture.head_hidden_dim,
            dropout=architecture.head_dropout,
        )
    model = RecapModel(
        tokenizer=tokenizer,
        mixture_of_experts=mixture_of_experts,
        aggregation=aggregation,
        prediction_head=prediction_head,
        routed_mask=routed_mask,
        expert_eligibility_mask=expert_eligibility_mask,
        expert_names=expert_names,
    )
    model.idp_short_names = tuple(short_names)
    return model


@dataclass
class RecapModelOutput:
    """Contains predictions and intermediate representations from RECAP.

    Attributes:
        predictions: Standardized MRI-age predictions with shape
            ``[batch_size, 1]`` for the default model.
        subject_embedding: Participant representation with shape
            ``[batch_size, model_dim]``.
        expert_tokens: Pooled shared and routed expert tokens before
            cross-expert contextualization.
        contextualized_expert_tokens: Expert tokens after cross-expert
            contextualization.
        expert_token_mask: Boolean validity mask for expert tokens.
        expert_mass: Routing mass received by each routed expert.
        router_logits: Unnormalized metadata-routing scores.
        router_probabilities: Dense metadata-routing probabilities.
        router_gates: Sparse top-k routing weights.
        expert_indices: Selected routed-expert indices for each IDP.
        expert_load: Normalized routing load for each routed expert.
        load_balance_loss: Auxiliary MoE load-balancing loss.
        router_z_loss: Auxiliary router-logit regularization loss.
        router_entropy: Mean dense router entropy over routed IDPs.
        feature_missing_mask: Resolved participant-level missing-value mask.
    """

    predictions: Tensor
    subject_embedding: Tensor
    expert_tokens: Tensor
    contextualized_expert_tokens: Tensor
    expert_token_mask: Tensor
    expert_mass: Tensor
    router_logits: Tensor
    router_probabilities: Tensor
    router_gates: Tensor
    expert_indices: Tensor
    expert_load: Tensor
    load_balance_loss: Tensor
    router_z_loss: Tensor
    router_entropy: Tensor
    router_anchor_loss: Tensor
    initial_assignment_agreement: Tensor
    routing_mode_code: Tensor
    feature_missing_mask: Tensor
    task_representations: Tensor | None = None
    task_expert_weights: Tensor | None = None
    expert_age_contributions: Tensor | None = None
    prediction_intercepts: Tensor | None = None


class RecapModel(nn.Module):
    """Metadata-conditioned mixture-of-experts model for brain-age prediction.

    Participant IDP values are fused with fixed IDP-description embeddings. Metadata-only routing then assigns routable
    IDPs to sparse experts, while a shared expert processes every IDP. Expert-level tokens are pooled and contextualized
    to form a participant representation used for MRI-age regression.
    """

    routed_mask: Tensor
    expert_eligibility_mask: Tensor
    idp_short_names: tuple[str, ...] | None = None

    def __init__(
        self,
        tokenizer: IDPTokenizer,
        mixture_of_experts: MetadataMixtureOfExperts,
        aggregation: ExpertAggregation,
        prediction_head: MultiTaskRegressionHead | TaskGatedRegressionHead | AdditiveExpertRegressionHead,
        routed_mask: Tensor,
        expert_eligibility_mask: Tensor,
        expert_names: tuple[str, ...],
    ) -> None:
        """Initializes RECAP from its model components.

        Args:
            tokenizer: IDP value and metadata tokenizer.
            mixture_of_experts: Shared and metadata-routed expert block.
            aggregation: Expert-token pooling and contextualization module.
            prediction_head: Multi-task regression head.
            routed_mask: Boolean tensor with shape ``[num_idps]``. ``False``
                excludes a global IDP from routed experts while retaining it
                in the shared expert.
            expert_eligibility_mask: Optional Boolean IDP-by-expert candidate
                mask. Defaults to all routed experts for routable IDPs.
            expert_names: Optional stable output name for each routed expert.

        Raises:
            ValueError: If component dimensions or the routed mask disagree.
        """
        super().__init__()

        if routed_mask.ndim != 1:
            raise ValueError("routed_mask must have shape [num_idps].")

        if routed_mask.shape[0] != tokenizer.num_idps:
            raise ValueError(
                "routed_mask must contain one value for each tokenizer IDP: "
                f"expected {tokenizer.num_idps}, received {routed_mask.shape[0]}."
            )

        if tokenizer.model_dim != mixture_of_experts.model_dim:
            raise ValueError("tokenizer and mixture_of_experts model dimensions must match.")

        if aggregation.token_aggregator.model_dim != tokenizer.model_dim:
            raise ValueError("tokenizer and aggregation model dimensions must match.")

        if aggregation.token_aggregator.num_experts != mixture_of_experts.num_routed_experts:
            raise ValueError("aggregation and mixture_of_experts must use the same number of routed experts.")

        if prediction_head.input_dim != tokenizer.model_dim:
            raise ValueError("prediction_head input_dim must equal the tokenizer model dimension.")

        router_metadata_dim = mixture_of_experts.router.metadata_dim
        if tokenizer.router_dim != router_metadata_dim:
            raise ValueError("tokenizer router_dim must equal the MoE router metadata dimension.")

        self.tokenizer = tokenizer
        self.mixture_of_experts = mixture_of_experts
        self.aggregation = aggregation
        self.prediction_head = prediction_head

        self.num_idps = tokenizer.num_idps
        self.model_dim = tokenizer.model_dim
        self.num_tasks = prediction_head.num_tasks

        self.register_buffer(
            "routed_mask",
            routed_mask.detach().clone().to(dtype=torch.bool),
            persistent=True,
        )
        expected_eligibility_shape = (tokenizer.num_idps, mixture_of_experts.num_routed_experts)
        if tuple(expert_eligibility_mask.shape) != expected_eligibility_shape:
            raise ValueError(f"expert_eligibility_mask must have shape {expected_eligibility_shape}.")
        self.register_buffer(
            "expert_eligibility_mask",
            expert_eligibility_mask.detach().clone().to(dtype=torch.bool),
            persistent=True,
        )
        if len(expert_names) != mixture_of_experts.num_routed_experts or len(set(expert_names)) != len(expert_names):
            raise ValueError("expert_names must contain one unique name per routed expert.")
        self.expert_names = expert_names

    @property
    def routing_mode(self) -> str:
        """Returns the current routing-training phase."""
        return self.mixture_of_experts.router.routing_mode

    def set_routing_mode(self, mode: Literal["fixed_initial", "adaptive"]) -> None:
        """Changes the routing-training phase."""
        self.mixture_of_experts.router.set_routing_mode(mode)

    def forward(
        self,
        idp_values: Tensor,
        feature_missing_mask: Tensor | None = None,
    ) -> RecapModelOutput:
        """Predicts standardized MRI age from participant IDP values.

        Args:
            idp_values: Standardized IDP measurements with shape
                ``[batch_size, num_idps]``. NaNs are treated as missing.
            feature_missing_mask: Optional Boolean mask with the same shape as
                ``idp_values``. ``True`` marks an originally missing value.

        Returns:
            Predictions, participant and expert representations, routing
            diagnostics, auxiliary losses, and the resolved missing mask.
        """
        if idp_values.ndim != 2:
            raise ValueError("idp_values must have shape [batch_size, num_idps].")

        if idp_values.shape[1] != self.num_idps:
            raise ValueError(
                "The number of IDP value columns does not match the tokenizer: "
                f"expected {self.num_idps}, received {idp_values.shape[1]}."
            )

        if feature_missing_mask is not None and feature_missing_mask.shape != idp_values.shape:
            raise ValueError("feature_missing_mask must have the same shape as idp_values.")

        tokenizer_output = self.tokenizer(
            values=idp_values,
            feature_missing_mask=feature_missing_mask,
        )

        feature_valid_mask = ~tokenizer_output.feature_missing_mask

        if torch.any(~feature_valid_mask.any(dim=1)):
            raise ValueError("Every participant must have at least one observed IDP value.")

        moe_output = self.mixture_of_experts(
            tokens=tokenizer_output.tokens,
            router_metadata=tokenizer_output.router_embeddings,
            routed_mask=self.routed_mask,
            expert_eligibility_mask=self.expert_eligibility_mask,
        )

        aggregation_output = self.aggregation(
            idp_tokens=moe_output.hidden_states,
            router_gates=moe_output.router_gates,
            feature_mask=feature_valid_mask,
            routed_mask=self.routed_mask,
            expert_eligibility_mask=self.expert_eligibility_mask,
        )

        task_representations: Tensor | None = None
        task_expert_weights: Tensor | None = None
        expert_age_contributions: Tensor | None = None
        prediction_intercepts: Tensor | None = None
        subject_embedding = aggregation_output.subject_embedding
        if isinstance(self.prediction_head, AdditiveExpertRegressionHead):
            (
                predictions,
                task_representations,
                task_expert_weights,
                expert_age_contributions,
                prediction_intercepts,
            ) = self.prediction_head(
                aggregation_output.expert_tokens,
                aggregation_output.expert_token_mask,
            )
            subject_embedding = task_representations.mean(dim=1)
        elif isinstance(self.prediction_head, TaskGatedRegressionHead):
            predictions, task_representations, task_expert_weights = self.prediction_head(
                aggregation_output.expert_tokens,
                aggregation_output.expert_token_mask,
            )
            subject_embedding = task_representations.mean(dim=1)
        else:
            predictions = self.prediction_head(subject_embedding)

        return RecapModelOutput(
            predictions=predictions,
            subject_embedding=subject_embedding,
            expert_tokens=aggregation_output.expert_tokens,
            contextualized_expert_tokens=aggregation_output.contextualized_tokens,
            expert_token_mask=aggregation_output.expert_token_mask,
            expert_mass=aggregation_output.expert_mass,
            router_logits=moe_output.router_logits,
            router_probabilities=moe_output.router_probabilities,
            router_gates=moe_output.router_gates,
            expert_indices=moe_output.expert_indices,
            expert_load=moe_output.expert_load,
            load_balance_loss=moe_output.load_balance_loss,
            router_z_loss=moe_output.router_z_loss,
            router_entropy=moe_output.router_entropy,
            router_anchor_loss=moe_output.router_anchor_loss,
            initial_assignment_agreement=moe_output.initial_assignment_agreement,
            routing_mode_code=moe_output.routing_mode_code,
            feature_missing_mask=tokenizer_output.feature_missing_mask,
            task_representations=task_representations,
            task_expert_weights=task_expert_weights,
            expert_age_contributions=expert_age_contributions,
            prediction_intercepts=prediction_intercepts,
        )
