from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class AggregationOutput:
    """Contains outputs produced by expert-token aggregation.

    Attributes:
        expert_tokens: Expert tokens before cross-expert contextualization.
            The shape is ``[batch_size, num_experts + 1, model_dim]``. The
            first token is the shared expert token.
        contextualized_tokens: Expert tokens after cross-expert encoding. The
            shape is ``[batch_size, num_experts + 1, model_dim]``.
        subject_embedding: Participant-level representation derived from the
            contextualized shared expert token. The shape is
            ``[batch_size, model_dim]``.
        expert_token_mask: Boolean mask indicating which expert tokens contain
            routed IDPs. The shape is
            ``[batch_size, num_experts + 1]``. The shared token is always
            marked as valid.
        expert_mass: Sum of routing weights assigned to each routed expert.
            The shape is ``[batch_size, num_experts]``.
    """

    expert_tokens: Tensor
    contextualized_tokens: Tensor
    subject_embedding: Tensor
    expert_token_mask: Tensor
    expert_mass: Tensor


class ExpertTokenAggregator(nn.Module):
    """Aggregates IDP tokens into shared and routed expert tokens.

    The shared token is calculated from all valid IDP tokens. Each routed
    expert token is calculated as a routing-weighted average of the IDP tokens
    assigned to that expert.

    Metadata-only routing can provide routing gates with shape
    ``[num_idps, num_experts]``. Participant-conditioned routing is also
    supported through routing gates with shape
    ``[batch_size, num_idps, num_experts]``.
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        dropout: float = 0.0,
        eps: float = 1e-8,
        shared_only_global: bool = False,
    ) -> None:
        """Initializes the expert-token aggregator.

        Args:
            model_dim: Dimension of each IDP and expert token.
            num_experts: Number of routed experts.
            dropout: Dropout applied after adding expert identity embeddings.
                Defaults to 0.0.
            eps: Small value used to prevent division by zero. Defaults to
                ``1e-8``.
            shared_only_global: Whether the shared token pools only IDPs
                excluded from regional routing.

        Raises:
            ValueError: If dimensions or hyperparameters are invalid.
        """
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.model_dim = model_dim
        self.num_experts = num_experts
        self.eps = eps
        self.shared_only_global = shared_only_global

        # Index 0 represents the shared expert. Indices 1..K represent routed
        # experts.
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts + 1, model_dim))

        self.output_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes learnable expert identity embeddings."""
        nn.init.normal_(
            self.expert_embeddings,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        idp_tokens: Tensor,
        router_gates: Tensor,
        feature_mask: Tensor | None = None,
        routed_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Aggregates participant-level IDP tokens into expert tokens.

        Args:
            idp_tokens: Participant-specific IDP tokens with shape
                ``[batch_size, num_idps, model_dim]``.
            router_gates: Routing weights with shape
                ``[num_idps, num_experts]`` or
                ``[batch_size, num_idps, num_experts]``.
            feature_mask: Optional boolean mask with shape
                ``[batch_size, num_idps]``. ``True`` indicates that an IDP
                token should participate in aggregation. When omitted, every
                IDP token participates.
            routed_mask: Optional boolean mask with shape ``[num_idps]``.
                ``True`` indicates that an IDP may enter routed experts.
                Global IDPs can be excluded by setting their values to
                ``False``. The shared token still receives every valid IDP.

        Returns:
            A tuple containing:

            - Expert tokens with shape
              ``[batch_size, num_experts + 1, model_dim]``.
            - Expert token validity mask with shape
              ``[batch_size, num_experts + 1]``.
            - Routing mass with shape
              ``[batch_size, num_experts]``.

        Raises:
            ValueError: If an input tensor has an unexpected shape.
        """
        if idp_tokens.ndim != 3:
            raise ValueError("idp_tokens must have shape [batch_size, num_idps, model_dim].")

        batch_size, num_idps, model_dim = idp_tokens.shape

        if model_dim != self.model_dim:
            raise ValueError(f"Expected IDP token dimension {self.model_dim}, but received {model_dim}.")

        gates = self._prepare_router_gates(
            router_gates=router_gates,
            batch_size=batch_size,
            num_idps=num_idps,
            device=idp_tokens.device,
            dtype=idp_tokens.dtype,
        )

        valid_feature_mask = self._prepare_feature_mask(
            feature_mask=feature_mask,
            batch_size=batch_size,
            num_idps=num_idps,
            device=idp_tokens.device,
        )

        valid_routed_mask = self._prepare_routed_mask(
            routed_mask=routed_mask,
            num_idps=num_idps,
            device=idp_tokens.device,
        )

        feature_weights = valid_feature_mask.to(dtype=idp_tokens.dtype)

        # The scientific-model variant restricts the shared token to global
        # IDPs; the compatibility default retains the original all-IDP pool.
        shared_weights = feature_weights
        if self.shared_only_global:
            shared_weights = shared_weights * (~valid_routed_mask)[None, :].to(dtype=idp_tokens.dtype)
        shared_mass = shared_weights.sum(
            dim=1,
            keepdim=True,
        )

        if torch.any(shared_mass == 0):
            raise ValueError("Every participant must have at least one valid IDP token.")

        shared_token = torch.einsum(
            "bp,bpd->bd",
            shared_weights,
            idp_tokens,
        )
        shared_token = shared_token / shared_mass.clamp_min(self.eps)

        # Routed experts receive only routable and valid IDPs.
        routed_weights = gates * feature_weights.unsqueeze(-1)
        routed_weights = routed_weights * valid_routed_mask[None, :, None].to(dtype=idp_tokens.dtype)

        expert_mass = routed_weights.sum(dim=1)

        routed_tokens = torch.einsum(
            "bpk,bpd->bkd",
            routed_weights,
            idp_tokens,
        )
        routed_tokens = routed_tokens / expert_mass.unsqueeze(-1).clamp_min(self.eps)

        expert_tokens = torch.cat(
            [
                shared_token.unsqueeze(1),
                routed_tokens,
            ],
            dim=1,
        )

        expert_tokens = expert_tokens + self.expert_embeddings.unsqueeze(0)
        expert_tokens = self.output_norm(expert_tokens)
        expert_tokens = self.dropout(expert_tokens)

        # The shared expert is always valid. A routed expert is valid only
        # when at least one IDP contributes positive routing mass.
        shared_token_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=idp_tokens.device,
        )
        routed_token_mask = expert_mass > self.eps

        expert_token_mask = torch.cat(
            [
                shared_token_mask,
                routed_token_mask,
            ],
            dim=1,
        )

        return expert_tokens, expert_token_mask, expert_mass

    def _prepare_router_gates(
        self,
        router_gates: Tensor,
        batch_size: int,
        num_idps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Validates and broadcasts routing gates."""
        if router_gates.ndim == 2:
            expected_shape = (num_idps, self.num_experts)

            if tuple(router_gates.shape) != expected_shape:
                raise ValueError(
                    f"Static router_gates must have shape {expected_shape}, but received {tuple(router_gates.shape)}."
                )

            router_gates = router_gates.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

        elif router_gates.ndim == 3:
            expected_shape = (
                batch_size,
                num_idps,
                self.num_experts,
            )

            if tuple(router_gates.shape) != expected_shape:
                raise ValueError(
                    f"Dynamic router_gates must have shape {expected_shape}, but received {tuple(router_gates.shape)}."
                )

        else:
            raise ValueError(
                "router_gates must have shape [num_idps, num_experts] or [batch_size, num_idps, num_experts]."
            )

        if torch.any(router_gates < 0):
            raise ValueError("router_gates must contain non-negative weights.")

        return router_gates.to(
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _prepare_feature_mask(
        feature_mask: Tensor | None,
        batch_size: int,
        num_idps: int,
        device: torch.device,
    ) -> Tensor:
        """Validates or creates a participant-level feature mask."""
        if feature_mask is None:
            return torch.ones(
                batch_size,
                num_idps,
                dtype=torch.bool,
                device=device,
            )

        expected_shape = (batch_size, num_idps)

        if tuple(feature_mask.shape) != expected_shape:
            raise ValueError(
                f"feature_mask must have shape {expected_shape}, but received {tuple(feature_mask.shape)}."
            )

        return feature_mask.to(
            device=device,
            dtype=torch.bool,
        )

    @staticmethod
    def _prepare_routed_mask(
        routed_mask: Tensor | None,
        num_idps: int,
        device: torch.device,
    ) -> Tensor:
        """Validates or creates the IDP routed-expert mask."""
        if routed_mask is None:
            return torch.ones(
                num_idps,
                dtype=torch.bool,
                device=device,
            )

        expected_shape = (num_idps,)

        if tuple(routed_mask.shape) != expected_shape:
            raise ValueError(f"routed_mask must have shape {expected_shape}, but received {tuple(routed_mask.shape)}.")

        return routed_mask.to(
            device=device,
            dtype=torch.bool,
        )


class CrossExpertEncoder(nn.Module):
    """Models interactions among shared and routed expert tokens.

    The shared expert token is expected at index zero. After contextualization, the token at index zero is used as the
    participant-level representation.
    """

    def __init__(
        self,
        model_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the cross-expert Transformer encoder.

        Args:
            model_dim: Dimension of each expert token.
            num_layers: Number of Transformer encoder layers. Defaults to 2.
            num_heads: Number of attention heads. Defaults to 4.
            feedforward_dim: Hidden dimension of each Transformer feedforward
                network. Defaults to ``4 * model_dim``.
            dropout: Dropout used by the Transformer. Defaults to 0.1.

        Raises:
            ValueError: If a model hyperparameter is invalid.
        """
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")

        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        if feedforward_dim is None:
            feedforward_dim = 4 * model_dim

        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")

        self.model_dim = model_dim
        self.num_layers = num_layers

        if num_layers == 0:
            self.encoder: nn.Module = nn.Identity()
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(model_dim),
            )

        self.subject_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        expert_tokens: Tensor,
        expert_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Contextualizes expert tokens and creates a subject embedding.

        Args:
            expert_tokens: Shared and routed expert tokens with shape
                ``[batch_size, num_experts + 1, model_dim]``.
            expert_token_mask: Optional boolean validity mask with shape
                ``[batch_size, num_experts + 1]``. ``True`` indicates that an
                expert token is valid.

        Returns:
            A tuple containing contextualized expert tokens and the
            participant-level subject embedding.

        Raises:
            ValueError: If input shapes are invalid or the shared token is
                masked.
        """
        if expert_tokens.ndim != 3:
            raise ValueError("expert_tokens must have shape [batch_size, num_expert_tokens, model_dim].")

        batch_size, num_expert_tokens, model_dim = expert_tokens.shape

        if model_dim != self.model_dim:
            raise ValueError(f"Expected expert token dimension {self.model_dim}, but received {model_dim}.")

        valid_mask: Tensor | None = None

        if expert_token_mask is not None:
            expected_shape = (
                batch_size,
                num_expert_tokens,
            )

            if tuple(expert_token_mask.shape) != expected_shape:
                raise ValueError(
                    "expert_token_mask must have shape "
                    f"{expected_shape}, but received "
                    f"{tuple(expert_token_mask.shape)}."
                )

            valid_mask = expert_token_mask.to(
                device=expert_tokens.device,
                dtype=torch.bool,
            )

            if not torch.all(valid_mask[:, 0]):
                raise ValueError("The shared expert token at index zero must always be valid.")

        if self.num_layers == 0:
            contextualized_tokens = self.encoder(expert_tokens)
        else:
            padding_mask = None if valid_mask is None else ~valid_mask

            contextualized_tokens = self.encoder(
                expert_tokens,
                src_key_padding_mask=padding_mask,
            )

        subject_embedding = self.subject_norm(contextualized_tokens[:, 0])

        return contextualized_tokens, subject_embedding


class ExpertAggregation(nn.Module):
    """Combines IDP pooling and cross-expert contextualization."""

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        eps: float = 1e-8,
        shared_only_global: bool = False,
    ) -> None:
        """Initializes the complete aggregation module.

        Args:
            model_dim: Dimension of IDP and expert tokens.
            num_experts: Number of routed experts.
            num_layers: Number of cross-expert Transformer layers.
            num_heads: Number of attention heads.
            feedforward_dim: Cross-expert feedforward dimension.
            dropout: Dropout used in aggregation and contextualization.
            eps: Small value used for stable weighted averaging.
            shared_only_global: Whether the shared token pools only global
                IDPs.
        """
        super().__init__()

        self.token_aggregator = ExpertTokenAggregator(
            model_dim=model_dim,
            num_experts=num_experts,
            dropout=dropout,
            eps=eps,
            shared_only_global=shared_only_global,
        )

        self.context_encoder = CrossExpertEncoder(
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        idp_tokens: Tensor,
        router_gates: Tensor,
        feature_mask: Tensor | None = None,
        routed_mask: Tensor | None = None,
        expert_eligibility_mask: Tensor | None = None,
    ) -> AggregationOutput:
        """Aggregates IDP tokens into a participant representation.

        Args:
            idp_tokens: IDP tokens with shape
                ``[batch_size, num_idps, model_dim]``.
            router_gates: Static or participant-specific routing weights.
            feature_mask: Optional participant-level valid-feature mask.
            routed_mask: Optional IDP-level mask controlling which IDPs may
                enter routed experts.
            expert_eligibility_mask: Optional IDP-by-expert Boolean mask used
                to verify that no routing mass crosses family boundaries.

        Returns:
            Aggregation outputs containing expert tokens, contextualized
            tokens, subject embeddings, expert masks, and routing masses.
        """
        if expert_eligibility_mask is not None:
            expected_shape = (idp_tokens.shape[1], self.token_aggregator.num_experts)
            if tuple(expert_eligibility_mask.shape) != expected_shape:
                raise ValueError(f"expert_eligibility_mask must have shape {expected_shape}.")
            eligibility = expert_eligibility_mask.to(device=router_gates.device, dtype=torch.bool)
            if router_gates.ndim == 2:
                forbidden_mass = router_gates.masked_select(~eligibility)
            elif router_gates.ndim == 3:
                forbidden_mass = router_gates.masked_select(~eligibility.unsqueeze(0))
            else:
                raise ValueError("router_gates must be two- or three-dimensional.")
            if torch.any(forbidden_mass != 0):
                raise ValueError("router_gates contain mass for family-ineligible experts.")

        (
            expert_tokens,
            expert_token_mask,
            expert_mass,
        ) = self.token_aggregator(
            idp_tokens=idp_tokens,
            router_gates=router_gates,
            feature_mask=feature_mask,
            routed_mask=routed_mask,
        )

        (
            contextualized_tokens,
            subject_embedding,
        ) = self.context_encoder(
            expert_tokens=expert_tokens,
            expert_token_mask=expert_token_mask,
        )

        return AggregationOutput(
            expert_tokens=expert_tokens,
            contextualized_tokens=contextualized_tokens,
            subject_embedding=subject_embedding,
            expert_token_mask=expert_token_mask,
            expert_mass=expert_mass,
        )
