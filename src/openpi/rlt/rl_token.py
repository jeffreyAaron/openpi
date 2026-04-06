"""RL Token encoder-decoder transformer for compressing VLA embeddings.

Implements the RL token extraction from Section IV-A of the RLT paper:
- Encoder g_phi: processes [z_1:M, e_rl] -> z_rl (Eq. 1)
- Decoder d_phi: autoregressively reconstructs z_bar from z_rl (Eq. 2)
- Module: combines both and computes reconstruction loss L_ro
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from openpi.rlt.config import RLTokenModelConfig


class RLTokenEncoder(nn.Module):
    """Encoder transformer that produces the RL token z_rl from VLA embeddings.

    Appends a learned e_rl embedding to the VLA token sequence, processes
    through a transformer encoder, and extracts the output at the RL token
    position (last token).

    Input:  z [B, M, vla_embed_dim] — image embeddings from frozen VLA
    Output: z_rl [B, rl_token_dim]  — compact RL state representation
    """

    def __init__(self, config: RLTokenModelConfig) -> None:
        super().__init__()
        self.config = config

        # Learned RL token embedding e_rl.
        self.e_rl = nn.Parameter(torch.randn(1, 1, config.vla_embed_dim) * 0.02)

        # Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vla_embed_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability.
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
        )

        # Project to rl_token_dim if different from vla_embed_dim.
        if config.rl_token_dim != config.vla_embed_dim:
            self.proj = nn.Linear(config.vla_embed_dim, config.rl_token_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, z: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Encode VLA embeddings into the RL token.

        Args:
            z: [B, M, vla_embed_dim] VLA image embeddings.
            pad_mask: [B, M] boolean mask (True = valid, False = padding).

        Returns:
            z_rl: [B, rl_token_dim] the RL token embedding.
        """
        B = z.shape[0]
        e_rl = self.e_rl.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat([z, e_rl], dim=1)  # [B, M+1, D]

        # Build key padding mask: True = IGNORE for nn.TransformerEncoder.
        if pad_mask is not None:
            rl_valid = torch.ones(B, 1, dtype=torch.bool, device=z.device)
            full_mask = torch.cat([pad_mask, rl_valid], dim=1)  # [B, M+1]
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None

        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        z_rl = out[:, -1, :]  # [B, D] — RL token is at last position.
        return self.proj(z_rl)


class RLTokenDecoder(nn.Module):
    """Decoder transformer that reconstructs VLA embeddings from z_rl.

    Autoregressively reconstructs the original VLA embeddings using
    teacher-forced inputs: [z_rl, z_bar_1, ..., z_bar_{M-1}] -> z_hat_1:M.
    Uses cross-attention from z_rl as memory.

    Input:  z_rl [B, rl_token_dim], z_bar [B, M, vla_embed_dim] (stop-grad targets)
    Output: z_hat [B, M, vla_embed_dim] (reconstructed embeddings)
    """

    def __init__(self, config: RLTokenModelConfig) -> None:
        super().__init__()
        self.config = config

        # Project rl_token_dim back to vla_embed_dim if needed.
        if config.rl_token_dim != config.vla_embed_dim:
            self.rl_proj = nn.Linear(config.rl_token_dim, config.vla_embed_dim)
        else:
            self.rl_proj = nn.Identity()

        # Decoder transformer with cross-attention to z_rl.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.vla_embed_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
        )

        # Linear output projection h_phi.
        self.head = nn.Linear(config.vla_embed_dim, config.vla_embed_dim)

    def forward(
        self, z_rl: Tensor, z_bar: Tensor, pad_mask: Tensor | None = None
    ) -> Tensor:
        """Reconstruct VLA embeddings from the RL token.

        Args:
            z_rl: [B, rl_token_dim] RL token.
            z_bar: [B, M, vla_embed_dim] stop-gradient VLA embeddings (targets).
            pad_mask: [B, M] boolean mask (True = valid).

        Returns:
            z_hat: [B, M, vla_embed_dim] reconstructed embeddings.
        """
        M = z_bar.shape[1]

        # Project z_rl to decoder input dimension.
        z_rl_proj = self.rl_proj(z_rl).unsqueeze(1)  # [B, 1, D]

        # Teacher-forced decoder input: [z_rl, z_bar_1, ..., z_bar_{M-1}].
        decoder_input = torch.cat([z_rl_proj, z_bar[:, :-1, :]], dim=1)  # [B, M, D]

        # Causal mask for autoregressive decoding.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_bar.device
        )

        # Use z_rl as cross-attention memory.
        memory = z_rl_proj  # [B, 1, D]

        # Build target key padding mask if needed.
        tgt_key_padding_mask = ~pad_mask if pad_mask is not None else None

        out = self.transformer(
            decoder_input,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.head(out)  # [B, M, D]


class RLTokenModule(nn.Module):
    """Combined RL token encoder-decoder for Stage 1 training.

    Encodes VLA embeddings into z_rl via the encoder, then trains the decoder
    to reconstruct the original embeddings. The reconstruction loss L_ro
    ensures z_rl retains enough information for downstream RL.
    """

    def __init__(self, config: RLTokenModelConfig) -> None:
        super().__init__()
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def forward(
        self, z: Tensor, pad_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: encode then decode, compute reconstruction loss.

        Args:
            z: [B, M, vla_embed_dim] VLA embeddings (will be detached for targets).
            pad_mask: [B, M] boolean mask (True = valid).

        Returns:
            z_rl: [B, rl_token_dim] the RL token.
            loss: scalar reconstruction loss L_ro.
        """
        z_bar = z.detach()  # Stop gradient on reconstruction targets.

        z_rl = self.encoder(z_bar, pad_mask)
        z_hat = self.decoder(z_rl, z_bar, pad_mask)

        # L_ro = E[ sum_i || z_hat_i - z_bar_i ||^2 ] (Eq. 2)
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()  # [B, M, 1]
            loss = ((z_hat - z_bar) ** 2 * mask).sum() / (mask.sum() * z_bar.shape[-1])
        else:
            loss = F.mse_loss(z_hat, z_bar)

        return z_rl, loss
