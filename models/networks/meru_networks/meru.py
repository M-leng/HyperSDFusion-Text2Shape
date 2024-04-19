# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from models.networks.meru_networks import lorentz as L
from models.networks.meru_networks.text_encoder import TransformerTextEncoder
from models.networks.meru_networks.image_encoder import build_timm_vit
import clip

class CLIPBaseline(nn.Module):
    """
    Our re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: torch.Tensor, project: bool, return_mid: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Project features to a unit hypersphere through L2 normalization.
        """

        # Truncate tokens that are longer than context_length:
        # for idx, inst_tokens in enumerate(tokens):
        #     if len(inst_tokens) > self.textual.context_length:
        #         eot_token = inst_tokens[-1]
        #         inst_tokens = inst_tokens[: self.textual.context_length]
        #         inst_tokens[-1] = eot_token
        #         tokens[idx] = inst_tokens
        #     if len(inst_tokens) < self.textual.context_length:
        #         inst_tokens
        #
        # # Pad all tokens on the right.
        # tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)
        if return_mid:
            return text_feats
        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        return image_feats, text_feats


class MERU(CLIPBaseline):
    """
    Our MERU model, that modifies CLIP to embed images and text in a hyperbolic
    space. Modifications are as follows:

    1. Lift embeddings from encoders onto the Lorentz hyperboloid using
       exponential map operator, instead of projecting via L2-normalization.

    2. Modify the contrastive loss to use the negative Lorentzian distance as
       a similarity measure instead of cosine similarity.

    3. Use a textual entailment loss to enforce a partial-order relationship
       between paired text and image embeddings (Note that an equivalent loss
       for CLIP is not mathematically defined).
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.curv.exp())

        return image_feats

    def encode_text(self, tokens: torch.Tensor, project: bool, return_mid: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False, return_mid=return_mid)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        return image_feats, text_feats


class MeruTextEncoder(nn.Module):
    def __init__(self, n_embed=512, max_seq_len=77):
        super().__init__()
        self.max_seq_len = max_seq_len
        img_encoder = build_timm_vit(
            arch="vit_base_patch16_224",
            global_pool="token",
            use_sincos2d_pos=True,
        )
        text_encoder = TransformerTextEncoder(
            arch="L12_W512", vocab_size=49408, context_length=max_seq_len
        )
        self.model = MERU(img_encoder, text_encoder, embed_dim=n_embed)
        state = torch.load('models/networks/meru_networks/meru_vit_b.pth')
        # model, optimizer, scheduler, scaler, iteration
        self.model.load_state_dict(state['model'])
    def forward(self, text):
        tokens = clip.tokenize(text,context_length=self.max_seq_len, truncate=True)
        text_feature = self.model.encode_text(tokens, project=False, return_mid=True)
        return text_feature

class MeruTransfer(nn.Module):
    def __init__(self, n_embed=512, num_heads=4, ff_size=2048, dropout=0., activation="gelu",
                        num_layers=2, n_layer=0):
        super().__init__()
        self.pre_proj = nn.Linear(512, n_embed)

        TransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.TransEncoder = nn.TransformerEncoder(
            TransEncoderLayer,
            num_layers=num_layers)
        self.text_ln = nn.LayerNorm(n_embed)
    def forward(self, text):
        x = text.permute(1, 0, 2)
        x = self.pre_proj(x)
        xf_out = self.TransEncoder(x)
        text_feature = self.text_ln(xf_out)
        text_feature = text_feature.permute(1, 0, 2)
        return text_feature

# img_encoder = build_timm_vit(
#         arch="vit_base_patch16_224",
#         global_pool="token",
#         use_sincos2d_pos=True,
#     )
# text_encoder = TransformerTextEncoder(
#         arch="L12_W512", vocab_size=49408, context_length=77
#     )
# model = MERU(img_encoder, text_encoder, embed_dim=512)
#
# state = torch.load('meru_vit_b.pth')
# #model, optimizer, scheduler, scaler, iteration
# model.load_state_dict(state['model'])
# text = ['this is a apple', 'this is a text, what is it']
#
# token_model = Tokenizer()
# tokens = token_model(text)
# print(tokens)
# text_feature = model.encode_text(tokens, project=False, return_mid=True)
# print(text_feature.shape)
# print("load ok")