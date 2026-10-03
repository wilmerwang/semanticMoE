from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ROUTING_MODES = {"fixed_initial": 0, "adaptive": 1}


@dataclass
class MoEOutput:
    """Contains outputs produced by the metadata-conditioned MoE block.

    Attributes:
        hidden_states: Output token representations with shape
            ``[batch_size, num_idps, model_dim]``.
        shared_output: Updates produced by the shared expert with shape
            ``[batch_size, num_idps, model_dim]``.
        routed_output: Weighted updates produced by routed experts with shape
            ``[batch_size, num_idps, model_dim]``.
        router_logits: Unnormalized routing scores with shape
            ``[num_idps, num_routed_experts]``.
        router_probabilities: Dense routing probabilities before top-k
            sparsification. Rows excluded by ``routed_mask`` contain zeros.
        router_gates: Sparse and normalized routing weights with shape
            ``[num_idps, num_routed_experts]``.
        expert_indices: Selected expert indices for each IDP with shape
            ``[num_idps, top_k]``. Excluded IDPs contain ``-1``.
        expert_load: Fraction of routed gate mass assigned to each expert.
        load_balance_loss: Auxiliary expert load-balancing loss.
        router_z_loss: Auxiliary loss controlling router logit magnitude.
        router_entropy: Mean entropy of routing probabilities over routed IDPs.
    """

    hidden_states: Tensor
    shared_output: Tensor
    routed_output: Tensor
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


@dataclass
class RouterOutput:
    """Contains outputs produced by the metadata router.

    Attributes:
        logits: Unnormalized routing scores.
        probabilities: Dense routing probabilities.
        gates: Sparse top-k routing gates.
        expert_indices: Selected expert indices.
        expert_load: Fraction of routed gate mass assigned to each expert.
        load_balance_loss: Auxiliary expert load-balancing loss.
        z_loss: Auxiliary router logit regularization loss.
        entropy: Mean router entropy.
    """

    logits: Tensor
    probabilities: Tensor
    gates: Tensor
    expert_indices: Tensor
    expert_load: Tensor
    load_balance_loss: Tensor
    z_loss: Tensor
    entropy: Tensor
    anchor_loss: Tensor
    initial_assignment_agreement: Tensor
    routing_mode_code: Tensor


def _build_activation(
    activation: Literal["gelu", "relu", "silu"],
) -> nn.Module:
    """Builds an activation module.

    Args:
        activation: Name of the activation function.

    Returns:
        The requested activation module.

    Raises:
        ValueError: If the activation name is unsupported.
    """
    if activation == "gelu":
        return nn.GELU()

    if activation == "relu":
        return nn.ReLU()

    if activation == "silu":
        return nn.SiLU()

    raise ValueError(f"activation must be one of {{'gelu', 'relu', 'silu'}}, but received {activation!r}.")


class FeedForwardExpert(nn.Module):
    """Implements a feed-forward expert operating on IDP tokens."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
    ) -> None:
        """Initializes the feed-forward expert.

        Args:
            model_dim: Input and output token dimension.
            hidden_dim: Hidden feed-forward dimension.
            dropout: Dropout probability. Defaults to 0.1.
            activation: Activation function name. Defaults to ``"gelu"``.

        Raises:
            ValueError: If a dimension is non-positive or dropout is invalid.
        """
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Applies the expert to IDP tokens.

        Args:
            tokens: Token tensor whose final dimension is ``model_dim``.
                Supported shapes include ``[batch_size, num_idps, model_dim]``
                and ``[batch_size, selected_idps, model_dim]``.

        Returns:
            Expert outputs with the same shape as ``tokens``.
        """
        return self.network(tokens)


class SharedExpert(FeedForwardExpert):
    """Feed-forward expert shared by all imaging-derived phenotypes."""


class RoutedExpert(FeedForwardExpert):
    """Feed-forward expert selected through metadata-conditioned routing."""


class MetadataRouter(nn.Module):
    """Routes IDPs to experts using only IDP metadata embeddings.

    The router does not use participant-specific IDP values. Consequently, the same IDP receives the same expert routing
    distribution for every participant, making expert specialization stable and interpretable.
    """

    initial_assignments: Tensor
    routing_mode_code: Tensor

    def __init__(
        self,
        metadata_dim: int,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        initial_assignments: Tensor | None = None,
        anchor_label_smoothing: float = 0.2,
    ) -> None:
        """Initializes the metadata-conditioned router.

        Args:
            metadata_dim: Dimension of each router metadata embedding.
            num_experts: Number of routed experts.
            top_k: Number of experts selected for each routed IDP. Defaults to
                2.
            hidden_dim: Optional hidden dimension for the router. When omitted,
                the router uses a single linear projection.
            temperature: Softmax temperature. Smaller values produce sharper
                routing distributions. Defaults to 1.0.
            dropout: Router hidden-layer dropout probability. Defaults to 0.0.
            initial_assignments: Precomputed expert index for every IDP. Use
                ``-1`` for shared-only global IDPs.
            anchor_label_smoothing: Probability mass assigned to non-target
                experts in the persistent semantic anchor objective.

        Raises:
            ValueError: If an argument is outside its valid range.
        """
        super().__init__()

        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive.")

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")

        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if not 0.0 <= anchor_label_smoothing < 1.0:
            raise ValueError("anchor_label_smoothing must be in the interval [0, 1).")

        self.metadata_dim = metadata_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.anchor_label_smoothing = anchor_label_smoothing

        if initial_assignments is None or initial_assignments.ndim != 1:
            raise ValueError("initial_assignments must have shape [num_idps].")
        if torch.any(initial_assignments >= num_experts) or torch.any(initial_assignments < -1):
            raise ValueError("initial_assignments contains an invalid expert index.")
        self.register_buffer(
            "initial_assignments",
            initial_assignments.detach().clone().to(dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "routing_mode_code",
            torch.tensor(ROUTING_MODES["adaptive"], dtype=torch.long),
            persistent=True,
        )
        if hidden_dim is None:
            self.network = nn.Sequential(
                nn.LayerNorm(metadata_dim),
                nn.Linear(metadata_dim, num_experts),
            )
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(metadata_dim),
                nn.Linear(metadata_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_experts),
            )

    @property
    def routing_mode(self) -> str:
        """Return the active routing phase name."""
        code = int(self.routing_mode_code.item())
        return next(name for name, value in ROUTING_MODES.items() if value == code)

    def set_routing_mode(self, mode: Literal["fixed_initial", "adaptive"]) -> None:
        """Select fixed initialization or adaptive routing."""
        if mode not in ROUTING_MODES:
            raise ValueError(f"Unsupported routing mode: {mode!r}")
        self.routing_mode_code.fill_(ROUTING_MODES[mode])

    def _learned_probabilities(
        self,
        metadata: Tensor,
        active_mask: Tensor,
        eligibility_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute family-constrained learned probabilities and logits."""
        logits = self.network(metadata)
        scaled_logits = logits / self.temperature
        safe_eligibility_mask = eligibility_mask.clone()
        safe_eligibility_mask[~active_mask] = True
        constrained_logits = scaled_logits.masked_fill(~safe_eligibility_mask, float("-inf"))
        return torch.softmax(constrained_logits, dim=-1), constrained_logits

    def _top_k_gates(self, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        """Sparsify learned probabilities to normalized top-k gates."""
        top_probabilities, expert_indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        top_gates = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(top_probabilities.dtype).eps
        )
        gates = torch.zeros_like(probabilities)
        gates.scatter_(dim=-1, index=expert_indices, src=top_gates)
        return gates, expert_indices

    def forward(
        self,
        metadata: Tensor,
        routed_mask: Tensor | None = None,
        expert_eligibility_mask: Tensor | None = None,
    ) -> RouterOutput:
        """Computes sparse top-k routing gates from IDP metadata.

        Args:
            metadata: Router metadata embeddings with shape
                ``[num_idps, metadata_dim]``.
            routed_mask: Optional boolean tensor with shape ``[num_idps]``.
                ``True`` indicates that an IDP may enter routed experts.
                ``False`` is appropriate for global IDPs that should use only
                the shared expert.
            expert_eligibility_mask: Optional Boolean tensor with shape
                ``[num_idps, num_experts]``. ``True`` marks experts that an
                IDP is allowed to enter.

        Returns:
            Metadata routing outputs.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if metadata.ndim != 2:
            raise ValueError("metadata must have shape [num_idps, metadata_dim].")

        if metadata.shape[-1] != self.metadata_dim:
            raise ValueError(
                "The final metadata dimension does not match metadata_dim: "
                f"expected {self.metadata_dim}, received {metadata.shape[-1]}."
            )

        num_idps = metadata.shape[0]
        device = metadata.device
        if num_idps != len(self.initial_assignments):
            raise ValueError("metadata does not match the routing initialization artifact.")

        if routed_mask is None:
            active_mask = torch.ones(
                num_idps,
                dtype=torch.bool,
                device=device,
            )
        else:
            if routed_mask.ndim != 1:
                raise ValueError("routed_mask must have shape [num_idps].")

            if routed_mask.shape[0] != num_idps:
                raise ValueError("routed_mask and metadata must contain the same number of IDPs.")

            active_mask = routed_mask.to(
                device=device,
                dtype=torch.bool,
            )

        if expert_eligibility_mask is None:
            eligibility_mask = torch.ones(
                num_idps,
                self.num_experts,
                dtype=torch.bool,
                device=device,
            )
        else:
            expected_shape = (num_idps, self.num_experts)
            if tuple(expert_eligibility_mask.shape) != expected_shape:
                raise ValueError(
                    "expert_eligibility_mask must have shape "
                    f"{expected_shape}, but received {tuple(expert_eligibility_mask.shape)}."
                )
            eligibility_mask = expert_eligibility_mask.to(device=device, dtype=torch.bool)

        eligible_counts = eligibility_mask.sum(dim=-1)
        if torch.any(active_mask & (eligible_counts < self.top_k)):
            raise ValueError("Every routed IDP must be eligible for at least top_k experts.")

        dense_probabilities, constrained_logits = self._learned_probabilities(
            metadata=metadata,
            active_mask=active_mask,
            eligibility_mask=eligibility_mask,
        )
        initial_assignments = self.initial_assignments.to(device=device)
        if torch.any(active_mask & (initial_assignments < 0)) or torch.any(~active_mask & (initial_assignments >= 0)):
            raise ValueError("initial_assignments and routed_mask disagree.")
        active_probabilities = dense_probabilities[active_mask]
        active_assignments = initial_assignments[active_mask]
        if active_mask.any():
            active_eligibility = eligibility_mask[active_mask]
            candidate_counts = active_eligibility.sum(dim=1, keepdim=True)
            non_target_counts = (candidate_counts - 1).clamp_min(1)
            anchor_targets = active_eligibility.to(active_probabilities.dtype)
            anchor_targets *= self.anchor_label_smoothing / non_target_counts
            target_probability = torch.where(
                candidate_counts > 1,
                active_probabilities.new_full(candidate_counts.shape, 1.0 - self.anchor_label_smoothing),
                active_probabilities.new_ones(candidate_counts.shape),
            )
            anchor_targets.scatter_(1, active_assignments.unsqueeze(-1), target_probability)
            anchor_loss = -torch.sum(
                anchor_targets * torch.log(active_probabilities.clamp_min(torch.finfo(active_probabilities.dtype).eps)),
                dim=1,
            ).mean()
        else:
            anchor_loss = self.network(metadata).sum() * 0.0

        mode = self.routing_mode
        if mode == "fixed_initial":
            gates = torch.zeros_like(dense_probabilities)
            active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            gates[active_rows, active_assignments] = 1.0
            expert_indices = torch.full((num_idps, self.top_k), -1, dtype=torch.long, device=device)
            expert_indices[active_rows, 0] = active_assignments
        elif mode == "adaptive":
            hard_gates, expert_indices = self._top_k_gates(dense_probabilities)
            # Straight-through routing: the forward pass is the same hard
            # Top-k computation used for evaluation, while gradients flow
            # through the dense metadata probabilities. With Top-1 this keeps
            # every IDP assigned to one interpretable expert.
            gates = hard_gates + dense_probabilities - dense_probabilities.detach() if self.training else hard_gates
        else:
            gates, expert_indices = self._top_k_gates(dense_probabilities)

        active_mask_float = active_mask.to(
            dtype=gates.dtype,
        ).unsqueeze(-1)

        probabilities = dense_probabilities * active_mask_float
        gates = gates * active_mask_float

        expert_indices = expert_indices.masked_fill(
            ~active_mask.unsqueeze(-1),
            -1,
        )
        routed_primary = gates[active_mask].argmax(dim=-1)
        initial_assignment_agreement = routed_primary.eq(active_assignments).to(gates.dtype).mean()

        (
            expert_load,
            load_balance_loss,
            z_loss,
            entropy,
        ) = self._compute_router_statistics(
            scaled_logits=constrained_logits,
            probabilities=dense_probabilities,
            gates=gates,
            active_mask=active_mask,
        )

        return RouterOutput(
            logits=constrained_logits,
            probabilities=probabilities,
            gates=gates,
            expert_indices=expert_indices,
            expert_load=expert_load,
            load_balance_loss=load_balance_loss,
            z_loss=z_loss,
            entropy=entropy,
            anchor_loss=anchor_loss,
            initial_assignment_agreement=initial_assignment_agreement,
            routing_mode_code=self.routing_mode_code.detach().clone(),
        )

    def _compute_router_statistics(
        self,
        scaled_logits: Tensor,
        probabilities: Tensor,
        gates: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes routing diagnostics and auxiliary losses.

        Args:
            scaled_logits: Temperature-scaled router logits.
            probabilities: Dense softmax probabilities.
            gates: Sparse routing gates after masking.
            active_mask: Boolean routed-IDP mask.

        Returns:
            Expert load, load-balancing loss, router z-loss, and entropy.
        """
        zero = scaled_logits.sum() * 0.0

        if not active_mask.any():
            expert_load = torch.zeros(
                self.num_experts,
                dtype=scaled_logits.dtype,
                device=scaled_logits.device,
            )
            return expert_load, zero, zero, zero

        active_probabilities = probabilities[active_mask]
        active_gates = gates[active_mask]
        active_logits = scaled_logits[active_mask]

        total_gate_mass = active_gates.sum().clamp_min(torch.finfo(active_gates.dtype).eps)

        expert_load = active_gates.sum(dim=0) / total_gate_mass

        # Hard assignment fractions are detached because argmax/Top-k is not
        # differentiable. Multiplication by the corresponding soft fractions
        # gives the Switch-style auxiliary objective: collapsed hard routing
        # creates a gradient that lowers probability for the overloaded expert.
        hard_assignments = active_gates.detach().gt(0).to(active_probabilities.dtype)
        selected_fraction = hard_assignments.mean(dim=0) / self.top_k
        probability_fraction = active_probabilities.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(selected_fraction * probability_fraction)

        log_partition = torch.logsumexp(
            active_logits,
            dim=-1,
        )
        z_loss = torch.mean(log_partition.square())

        epsilon = torch.finfo(active_probabilities.dtype).eps

        entropy = -torch.sum(
            active_probabilities * torch.log(active_probabilities.clamp_min(epsilon)),
            dim=-1,
        ).mean()

        return (
            expert_load,
            load_balance_loss,
            z_loss,
            entropy,
        )


class MetadataMixtureOfExperts(nn.Module):
    """Applies shared and metadata-routed experts to IDP tokens.

    Every IDP is processed by the shared expert. IDPs enabled by
    ``routed_mask`` are additionally processed by their selected routed
    experts. Routed expert outputs are combined using metadata-derived gates.

    Routed computation uses sparse dispatch: each routed expert receives only
    the IDP positions for which its routing gate is non-zero.
    """

    def __init__(
        self,
        model_dim: int,
        router_metadata_dim: int,
        expert_hidden_dim: int,
        num_routed_experts: int = 8,
        top_k: int = 2,
        router_hidden_dim: int | None = None,
        router_temperature: float = 1.0,
        expert_dropout: float = 0.1,
        router_dropout: float = 0.0,
        output_dropout: float = 0.1,
        shared_only_global: bool = False,
        initial_assignments: Tensor | None = None,
        anchor_label_smoothing: float = 0.2,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
    ) -> None:
        """Initializes the metadata-conditioned mixture-of-experts block.

        Args:
            model_dim: IDP token dimension.
            router_metadata_dim: Dimension of router metadata embeddings.
            expert_hidden_dim: Hidden dimension of each feed-forward expert.
            num_routed_experts: Number of routed experts. Defaults to 8.
            top_k: Number of routed experts selected for each IDP. Defaults to
                2.
            router_hidden_dim: Optional hidden dimension of the metadata
                router.
            router_temperature: Softmax temperature used by the router.
                Defaults to 1.0.
            expert_dropout: Dropout probability inside each expert. Defaults to
                0.1.
            router_dropout: Dropout probability inside the metadata router.
                Defaults to 0.0.
            output_dropout: Dropout applied to the combined expert update.
                Defaults to 0.1.
            shared_only_global: Whether shared-expert updates are restricted
                to IDPs excluded from regional routing.
            initial_assignments: Precomputed expert index for every IDP. Use
                ``-1`` for shared-only global IDPs.
            anchor_label_smoothing: Probability mass assigned to non-target
                experts in the persistent semantic anchor objective.
            activation: Expert activation function. Defaults to ``"gelu"``.

        Raises:
            ValueError: If output dropout is invalid.
        """
        super().__init__()

        if not 0.0 <= output_dropout < 1.0:
            raise ValueError("output_dropout must be in the interval [0, 1).")

        self.model_dim = model_dim
        self.num_routed_experts = num_routed_experts
        self.shared_only_global = shared_only_global

        self.input_normalization = nn.LayerNorm(model_dim)

        self.shared_expert = SharedExpert(
            model_dim=model_dim,
            hidden_dim=expert_hidden_dim,
            dropout=expert_dropout,
            activation=activation,
        )

        self.routed_experts = nn.ModuleList(
            RoutedExpert(
                model_dim=model_dim,
                hidden_dim=expert_hidden_dim,
                dropout=expert_dropout,
                activation=activation,
            )
            for _ in range(num_routed_experts)
        )

        self.router = MetadataRouter(
            metadata_dim=router_metadata_dim,
            num_experts=num_routed_experts,
            top_k=top_k,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            dropout=router_dropout,
            initial_assignments=initial_assignments,
            anchor_label_smoothing=anchor_label_smoothing,
        )

        self.output_dropout = nn.Dropout(output_dropout)

    def forward(
        self,
        tokens: Tensor,
        router_metadata: Tensor,
        routed_mask: Tensor | None = None,
        expert_eligibility_mask: Tensor | None = None,
    ) -> MoEOutput:
        """Applies shared and routed experts to IDP tokens.

        Args:
            tokens: Participant-specific fused IDP tokens with shape
                ``[batch_size, num_idps, model_dim]``.
            router_metadata: IDP metadata embeddings used only for routing,
                with shape ``[num_idps, router_metadata_dim]``.
            routed_mask: Optional boolean tensor with shape ``[num_idps]``.
                Global IDPs should have value ``False`` so they are processed
                only by the shared expert.
            expert_eligibility_mask: Optional Boolean IDP-by-expert candidate mask.

        Returns:
            Mixture-of-experts outputs and routing diagnostics.

        Raises:
            ValueError: If token or metadata shapes are inconsistent.
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch_size, num_idps, model_dim].")

        if tokens.shape[-1] != self.model_dim:
            raise ValueError(
                "The final token dimension does not match model_dim: "
                f"expected {self.model_dim}, received {tokens.shape[-1]}."
            )

        if router_metadata.ndim != 2:
            raise ValueError("router_metadata must have shape [num_idps, router_metadata_dim].")

        if tokens.shape[1] != router_metadata.shape[0]:
            raise ValueError("tokens and router_metadata must contain the same number of IDPs.")

        normalized_tokens = self.input_normalization(tokens)

        shared_output = self.shared_expert(normalized_tokens)

        if self.shared_only_global:
            if routed_mask is None:
                raise ValueError("routed_mask is required when shared_only_global is enabled.")
            global_mask = ~routed_mask.to(device=tokens.device, dtype=torch.bool)
            if not global_mask.any():
                raise ValueError("shared_only_global requires at least one global IDP.")
            shared_output = shared_output * global_mask[None, :, None].to(dtype=shared_output.dtype)

        router_output = self.router(
            metadata=router_metadata,
            routed_mask=routed_mask,
            expert_eligibility_mask=expert_eligibility_mask,
        )

        routed_output = self._apply_routed_experts(
            tokens=normalized_tokens,
            gates=router_output.gates,
        )

        combined_update = shared_output + routed_output

        hidden_states = tokens + self.output_dropout(combined_update)

        return MoEOutput(
            hidden_states=hidden_states,
            shared_output=shared_output,
            routed_output=routed_output,
            router_logits=router_output.logits,
            router_probabilities=router_output.probabilities,
            router_gates=router_output.gates,
            expert_indices=router_output.expert_indices,
            expert_load=router_output.expert_load,
            load_balance_loss=router_output.load_balance_loss,
            router_z_loss=router_output.z_loss,
            router_entropy=router_output.entropy,
            router_anchor_loss=router_output.anchor_loss,
            initial_assignment_agreement=router_output.initial_assignment_agreement,
            routing_mode_code=router_output.routing_mode_code,
        )

    def _apply_routed_experts(
        self,
        tokens: Tensor,
        gates: Tensor,
    ) -> Tensor:
        """Dispatches selected IDP tokens to routed experts.

        Args:
            tokens: Normalized IDP tokens with shape
                ``[batch_size, num_idps, model_dim]``.
            gates: Sparse routing gates with shape
                ``[num_idps, num_routed_experts]``.

        Returns:
            Weighted routed-expert output with the same shape as ``tokens``.
        """
        routed_output = torch.zeros_like(tokens)

        for expert_index, expert in enumerate(self.routed_experts):
            expert_gates = gates[:, expert_index]

            selected_positions = torch.nonzero(
                expert_gates > 0,
                as_tuple=False,
            ).squeeze(-1)

            if selected_positions.numel() == 0:
                continue

            expert_inputs = tokens.index_select(
                dim=1,
                index=selected_positions,
            )

            expert_outputs = expert(expert_inputs)

            selected_gates = expert_gates.index_select(
                dim=0,
                index=selected_positions,
            )

            weighted_outputs = expert_outputs * selected_gates.view(
                1,
                -1,
                1,
            )

            routed_output = routed_output.index_add(
                dim=1,
                index=selected_positions,
                source=weighted_outputs,
            )

        return routed_output
