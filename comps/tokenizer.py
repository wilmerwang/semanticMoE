from typing import NamedTuple

import torch
from torch import Tensor, nn


class IDPTokenizerOutput(NamedTuple):
    """Outputs produced by the imaging-derived phenotype tokenizer.

    Attributes:
        tokens: Participant-specific imaging-derived phenotype tokens with
            shape ``[batch_size, num_idps, model_dim]``.
        metadata_tokens: Metadata representations used to construct the final
            tokens, with shape ``[num_idps, model_dim]``.
        router_embeddings: Metadata-only representations intended for expert
            routing, with shape ``[num_idps, router_dim]``.
        value_embeddings: Participant-specific value representations with
            shape ``[batch_size, num_idps, model_dim]``.
        feature_missing_mask: Boolean mask with shape
            ``[batch_size, num_idps]``. A value of ``True`` indicates that the
            corresponding imaging-derived phenotype value was missing.
    """

    tokens: Tensor
    metadata_tokens: Tensor
    router_embeddings: Tensor
    value_embeddings: Tensor
    feature_missing_mask: Tensor


class MetadataEncoder(nn.Module):
    """Encodes fixed imaging-derived phenotype metadata embeddings.

    A shared metadata backbone generates two representations:

    1. A token representation that is fused with participant-specific values.
    2. A semantic router representation used to assign imaging-derived
       phenotypes to routed experts.

    An optional learnable imaging-derived phenotype identity embedding is added
    only to the token representation. It is intentionally excluded from the
    router representation so that routing remains determined by semantic
    metadata rather than arbitrary feature identifiers.
    """

    def __init__(
        self,
        metadata_input_dim: int,
        model_dim: int,
        router_dim: int,
        num_idps: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_idp_identity_embedding: bool = True,
    ) -> None:
        """Initializes the metadata encoder.

        Args:
            metadata_input_dim: Dimension of the pretrained text embeddings.
            model_dim: Dimension of the participant-specific IDP tokens.
            router_dim: Dimension of the metadata representations supplied to
                the expert router.
            num_idps: Number of imaging-derived phenotypes.
            hidden_dim: Hidden dimension of the shared metadata backbone.
                Defaults to 256.
            dropout: Dropout probability. Defaults to 0.1.
            use_idp_identity_embedding: Whether to add a learnable
                feature-identity embedding to the token metadata branch.
                Defaults to ``True``.

        Raises:
            ValueError: If a dimension is not positive or if dropout is outside
                the interval from zero to one.
        """
        super().__init__()

        dimensions = {
            "metadata_input_dim": metadata_input_dim,
            "model_dim": model_dim,
            "router_dim": router_dim,
            "num_idps": num_idps,
            "hidden_dim": hidden_dim,
        }

        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.num_idps = num_idps
        self.model_dim = model_dim
        self.router_dim = router_dim

        self.backbone = nn.Sequential(
            nn.LayerNorm(metadata_input_dim),
            nn.Linear(metadata_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.token_projection = nn.Linear(
            hidden_dim,
            model_dim,
        )
        self.router_projection = nn.Linear(
            hidden_dim,
            router_dim,
        )

        self.token_normalization = nn.LayerNorm(model_dim)
        self.router_normalization = nn.LayerNorm(router_dim)

        self.idp_identity_embedding: nn.Embedding | None

        if use_idp_identity_embedding:
            self.idp_identity_embedding = nn.Embedding(
                num_embeddings=num_idps,
                embedding_dim=model_dim,
            )
            nn.init.normal_(
                self.idp_identity_embedding.weight,
                mean=0.0,
                std=0.02,
            )
        else:
            self.idp_identity_embedding = None

    def forward(
        self,
        metadata_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encodes pretrained metadata embeddings.

        Args:
            metadata_embeddings: Pretrained description embeddings with shape
                ``[num_idps, metadata_input_dim]``.

        Returns:
            A tuple containing token metadata embeddings with shape
            ``[num_idps, model_dim]`` and router embeddings with shape
            ``[num_idps, router_dim]``.

        Raises:
            ValueError: If the input is not two-dimensional, contains an
                unexpected number of imaging-derived phenotypes, or contains
                non-finite values.
        """
        if metadata_embeddings.ndim != 2:
            raise ValueError("metadata_embeddings must have shape [num_idps, metadata_input_dim].")

        if metadata_embeddings.shape[0] != self.num_idps:
            raise ValueError(
                "The number of metadata embeddings does not match num_idps: "
                f"expected {self.num_idps}, "
                f"received {metadata_embeddings.shape[0]}."
            )

        if not torch.isfinite(metadata_embeddings).all():
            raise ValueError("metadata_embeddings contains non-finite values.")

        hidden = self.backbone(metadata_embeddings)

        metadata_tokens = self.token_projection(hidden)

        if self.idp_identity_embedding is not None:
            metadata_tokens = metadata_tokens + self.idp_identity_embedding.weight

        metadata_tokens = self.token_normalization(metadata_tokens)

        router_embeddings = self.router_normalization(self.router_projection(hidden))

        return metadata_tokens, router_embeddings


class ValueEncoder(nn.Module):
    """Encodes participant-specific imaging-derived phenotype values.

    The same multilayer perceptron is shared across all imaging-derived phenotypes. Feature identity is supplied
    separately by the metadata encoder.

    The value encoder receives both the standardized numerical value and a missing-value indicator. Missing values are
    replaced with zero before encoding, while the missing indicator informs the model that zero is a placeholder rather
    than an observed measurement.
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the value encoder.

        Args:
            model_dim: Output token dimension.
            hidden_dim: Hidden dimension of the value multilayer perceptron.
                Defaults to 64.
            dropout: Dropout probability. Defaults to 0.1.

        Raises:
            ValueError: If a dimension is not positive or if dropout is outside
                the interval from zero to one.
        """
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        # Each input contains:
        # 1. The standardized numerical value.
        # 2. A missing-value indicator.
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        values: Tensor,
        feature_missing_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encodes participant-specific numerical values.

        Args:
            values: Numerical imaging-derived phenotype values with shape
                ``[batch_size, num_idps]``. Values should normally already be
                standardized using training-set statistics.
            feature_missing_mask: Optional Boolean mask with the same shape as
                ``values``. ``True`` indicates an originally missing value.
                When omitted, missingness is inferred from ``NaN`` values.

        Returns:
            A tuple containing value embeddings with shape
            ``[batch_size, num_idps, model_dim]`` and the resolved Boolean
            missing-value mask with shape ``[batch_size, num_idps]``.

        Raises:
            ValueError: If the values are not two-dimensional, the mask has an
                incompatible shape, or infinite values are present.
        """
        if values.ndim != 2:
            raise ValueError("values must have shape [batch_size, num_idps].")

        values = values.to(dtype=torch.float32)

        if torch.isinf(values).any():
            raise ValueError("values contains positive or negative infinity.")

        nan_mask = torch.isnan(values)

        if feature_missing_mask is None:
            resolved_missing_mask = nan_mask
        else:
            if feature_missing_mask.shape != values.shape:
                raise ValueError("feature_missing_mask must have the same shape as values.")

            resolved_missing_mask = (
                feature_missing_mask.to(
                    device=values.device,
                    dtype=torch.bool,
                )
                | nan_mask
            )

        # A standardized value of zero is a neutral placeholder. The second
        # input channel tells the model whether that value was originally
        # missing.
        safe_values = torch.where(
            resolved_missing_mask,
            torch.zeros_like(values),
            values,
        )

        encoder_input = torch.stack(
            (
                safe_values,
                resolved_missing_mask.to(dtype=values.dtype),
            ),
            dim=-1,
        )

        value_embeddings = self.encoder(encoder_input)

        return value_embeddings, resolved_missing_mask


class IDPTokenizer(nn.Module):
    """Combines fixed metadata and participant-specific IDP measurements.

    Pretrained text embeddings are stored once and projected by the metadata encoder. For each participant, the
    resulting metadata token is added to the corresponding numerical value embedding.

    The router embedding remains metadata-only and is therefore identical across participants.
    """

    def __init__(
        self,
        metadata_embeddings: Tensor,
        model_dim: int = 128,
        router_dim: int = 128,
        metadata_hidden_dim: int = 256,
        value_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_idp_identity_embedding: bool = True,
        persist_metadata_embeddings: bool = True,
    ) -> None:
        """Initializes the imaging-derived phenotype tokenizer.

        Args:
            metadata_embeddings: Frozen pretrained description embeddings with
                shape ``[num_idps, metadata_input_dim]``.
            model_dim: Dimension of the final IDP tokens. Defaults to 128.
            router_dim: Dimension of metadata router embeddings. Defaults to
                128.
            metadata_hidden_dim: Hidden dimension of the metadata encoder.
                Defaults to 256.
            value_hidden_dim: Hidden dimension of the value encoder. Defaults
                to 64.
            dropout: Dropout probability. Defaults to 0.1.
            use_idp_identity_embedding: Whether to add a learnable IDP identity
                embedding to the token branch. The identity embedding is not
                used by the router. Defaults to ``True``.
            persist_metadata_embeddings: Whether the frozen pretrained
                metadata embeddings should be included in the module state
                dictionary. Defaults to ``True``.

        Raises:
            ValueError: If metadata embeddings do not have two dimensions or
                contain no imaging-derived phenotypes.
        """
        super().__init__()

        if metadata_embeddings.ndim != 2:
            raise ValueError("metadata_embeddings must have shape [num_idps, metadata_input_dim].")

        if metadata_embeddings.shape[0] == 0:
            raise ValueError("metadata_embeddings must contain at least one IDP.")

        if metadata_embeddings.shape[1] == 0:
            raise ValueError("metadata_embeddings must contain at least one dimension.")

        metadata_embeddings = metadata_embeddings.detach().clone().to(dtype=torch.float32)

        if not torch.isfinite(metadata_embeddings).all():
            raise ValueError("metadata_embeddings contains non-finite values.")

        self.num_idps = metadata_embeddings.shape[0]
        self.metadata_input_dim = metadata_embeddings.shape[1]
        self.model_dim = model_dim
        self.router_dim = router_dim

        self.register_buffer(
            "_metadata_embeddings",
            metadata_embeddings,
            persistent=persist_metadata_embeddings,
        )

        self.metadata_encoder = MetadataEncoder(
            metadata_input_dim=self.metadata_input_dim,
            model_dim=model_dim,
            router_dim=router_dim,
            num_idps=self.num_idps,
            hidden_dim=metadata_hidden_dim,
            dropout=dropout,
            use_idp_identity_embedding=(use_idp_identity_embedding),
        )

        self.value_encoder = ValueEncoder(
            model_dim=model_dim,
            hidden_dim=value_hidden_dim,
            dropout=dropout,
        )

        self.fusion_normalization = nn.LayerNorm(model_dim)
        self.fusion_dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: Tensor,
        feature_missing_mask: Tensor | None = None,
    ) -> IDPTokenizerOutput:
        """Converts participant IDP values into IDP tokens.

        Args:
            values: Participant-specific imaging-derived phenotype values with
                shape ``[batch_size, num_idps]``.
            feature_missing_mask: Optional mask with the same shape as
                ``values``. ``True`` indicates an originally missing value.

        Returns:
            Tokenizer outputs containing fused IDP tokens, metadata
            representations, value representations, router representations,
            and the resolved feature missing-value mask.

        Raises:
            ValueError: If the number of IDP value columns differs from the
                number of metadata embeddings.
        """
        if values.ndim != 2:
            raise ValueError("values must have shape [batch_size, num_idps].")

        if values.shape[1] != self.num_idps:
            raise ValueError(
                "The number of value columns does not match the number of "
                f"metadata embeddings: expected {self.num_idps}, "
                f"received {values.shape[1]}."
            )

        metadata_embeddings = self.get_buffer("_metadata_embeddings")

        metadata_tokens, router_embeddings = self.metadata_encoder(metadata_embeddings)

        value_embeddings, resolved_missing_mask = self.value_encoder(
            values=values,
            feature_missing_mask=feature_missing_mask,
        )

        # Equivalent to the dual-encoder additive fusion used by gene
        # expression models: fixed IDP semantics plus participant-specific
        # measurement information.
        tokens = value_embeddings + metadata_tokens.unsqueeze(0)

        tokens = self.fusion_normalization(tokens)
        tokens = self.fusion_dropout(tokens)

        return IDPTokenizerOutput(
            tokens=tokens,
            metadata_tokens=metadata_tokens,
            router_embeddings=router_embeddings,
            value_embeddings=value_embeddings,
            feature_missing_mask=resolved_missing_mask,
        )
