"""Wrapper around frozen PI0Pytorch for RL token extraction and reference actions.

Provides clean methods to:
1. Extract image embeddings from the VLA's prefix encoder (for RL token input).
2. Sample reference action chunks from the VLA's diffusion head.
3. Compute the full RL token z_rl by chaining extraction with the RL token encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.rlt.rl_token import RLTokenEncoder


class VLAEmbeddingExtractor:
    """Wraps a frozen PI0Pytorch model for embedding extraction.

    The VLA is never modified — all calls are under torch.no_grad().
    This class provides the interface between the VLA and the RL token system.

    Args:
        vla_model: A PI0Pytorch model (should be frozen / in eval mode).
        device: Device for tensor operations.
    """

    def __init__(self, vla_model: PI0Pytorch, device: torch.device) -> None:
        self.vla = vla_model
        self.device = device
        self.vla.eval()

    @torch.no_grad()
    def extract_image_embeddings(
        self, observation: Observation,
    ) -> tuple[Tensor, Tensor]:
        """Extract image token embeddings from the VLA's prefix encoder.

        Calls embed_prefix and returns only the image portion (excluding
        language tokens), since the paper drops language embeddings for the
        RL token (each task has a fixed instruction).

        Args:
            observation: Batched Observation with images, masks, prompts, state.

        Returns:
            img_embs: [B, M_img, 2048] image token embeddings.
            img_pad_mask: [B, M_img] boolean validity mask.
        """
        images, img_masks, lang_tokens, lang_masks, _state = (
            self.vla._preprocess_observation(observation, train=False)  # noqa: SLF001
        )

        # embed_prefix returns concatenated [image_tokens, language_tokens].
        # We need to compute how many image tokens there are to split them out.
        num_img_tokens = 0
        img_embs_list = []
        img_pad_masks_list = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.vla.paligemma_with_expert.embed_image(img)
            bsize, n_tokens = img_emb.shape[:2]
            num_img_tokens += n_tokens
            img_embs_list.append(img_emb)
            img_pad_masks_list.append(img_mask[:, None].expand(bsize, n_tokens))

        img_embs = torch.cat(img_embs_list, dim=1)  # [B, M_img, 2048]
        img_pad_mask = torch.cat(img_pad_masks_list, dim=1)  # [B, M_img]

        return img_embs, img_pad_mask

    @torch.no_grad()
    def sample_reference_actions(self, observation: Observation) -> Tensor:
        """Sample a full action chunk from the VLA's diffusion head.

        Args:
            observation: Batched Observation.

        Returns:
            actions: [B, H, action_dim] VLA-predicted action chunk (H=50 typically).
        """
        return self.vla.sample_actions(self.device, observation)

    @torch.no_grad()
    def extract_rl_token(
        self, observation: Observation, encoder: RLTokenEncoder,
    ) -> Tensor:
        """Convenience: extract embeddings + encode to z_rl in one call.

        Args:
            observation: Batched Observation.
            encoder: Frozen RLTokenEncoder.

        Returns:
            z_rl: [B, rl_token_dim] the RL token.
        """
        img_embs, img_pad_mask = self.extract_image_embeddings(observation)
        return encoder(img_embs, img_pad_mask)
