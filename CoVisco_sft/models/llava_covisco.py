"""llava_covisco.py - Top-level model.

Composition: ViT + TokenSelector + Projector + LLM (Qwen3).
forward() concatenates query / vit tokens per the dynamic strategy and injects them into the LLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import LLMConfig, ModelConfig, ProjectorConfig, TokenSelectorConfig, TokenStrategyConfig, ViTConfig
from .covisco_vit import CoViscoViT
from .projector import TwoLayerMLPProjector
from .token_selector import LearnableTokenSelector


class LlavaCoViscoModel(nn.Module):
    """CoVisco ViT + Qwen3 LLM joint model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ViT
        from ._covisco_encoder_src import CoViscoEncoderConfig
        vit_cfg = CoViscoEncoderConfig(
            hidden_size=config.vit.hidden_size,
            intermediate_size=config.vit.intermediate_size,
            num_hidden_layers=config.vit.num_layers,
            num_attention_heads=config.vit.num_attention_heads,
            num_channels=config.vit.num_channels,
            image_size=config.vit.image_size,
            patch_size=config.vit.patch_size,
            num_query_per_seg=config.vit.num_query_per_seg,
            segment_t_size=config.vit.segment_t_size,
            rope_theta=config.vit.rope_theta,
            layer_norm_eps=config.vit.layer_norm_eps,
            initializer_range=config.vit.initializer_range,
        )
        self.vit = CoViscoViT(vit_cfg)
        self.vit_hidden_size = config.vit.hidden_size
        self.num_query_per_seg = config.vit.num_query_per_seg

        # Load ViT pretrained weights
        if config.vit.pretrained_path:
            from .covisco_vit_checkpoint import load_vit_weights_direct
            print(f"[ViT] loading pretrained weights from {config.vit.pretrained_path}")
            report = load_vit_weights_direct(self.vit.encoder, config.vit.pretrained_path)
            print(f"[ViT] loaded {len(report['loaded'])} / {report['total_src']} source tensors")
            if report["missing"]:
                print(f"[ViT] missing keys ({len(report['missing'])}): {report['missing'][:20]}")
            if report["unexpected"]:
                print(f"[ViT] unexpected keys ({len(report['unexpected'])}): {report['unexpected'][:20]}")

        # Token Selector
        self.token_selector = LearnableTokenSelector(
            hidden_size=self.vit_hidden_size,
            num_layers=config.token_selector.num_layers,
            num_heads=config.token_selector.num_heads,
            selection_mode=config.token_selector.method,
            score_activation=config.token_selector.score_activation,
            gumbel_temperature=config.token_selector.gumbel_temperature,
            mmr_lambda=config.token_selector.mmr_lambda,
            logit_scale=config.token_selector.logit_scale,
        )

        # Projector
        self.projector = TwoLayerMLPProjector(
            vision_hidden=self.vit_hidden_size,
            llm_hidden=config.llm.hidden_size,
            sms=config.projector.sms,
            ffn_mult=config.projector.ffn_mult,
        )

        # LLM (lazy load: skip when path is empty, useful for mocks / unit tests)
        if config.llm.path:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm.path,
                torch_dtype=torch.bfloat16,
                output_loading_info=False,
            )
            print(f"[LLM] loaded from {config.llm.path}")
        else:
            self.language_model = None

        # Cache token ids
        self.image_pad_token_id = config.llm.image_pad_token_id
        self.vision_start_token_id = config.llm.vision_start_token_id
        self.vision_end_token_id = config.llm.vision_end_token_id

        # Validate the hardcoded vision token ids against the real tokenizer: these ids
        # drive masked_scatter of vision embeddings into <|image_pad|> and other
        # placeholder positions. If they disagree with the tokenizer's actual vocab,
        # vision features get written to the wrong tokens without any error. When the
        # LLM is loaded, also load the tokenizer from the same path and assert
        # consistency so config mistakes surface at startup.
        if config.llm.path:
            self._validate_vision_token_ids(config.llm.path)

        # Sparsity regularization coefficient (0 disables it)
        self.sparsity_lambda: float = getattr(config, "sparsity_lambda", 0.0)
        # The training loop uses it to separate sparsity from CE, avoiding multiplication by the label token count.
        self._last_sparsity_loss_tensor = None
        # First-frame token ratio cap during inference (0.0 = unlimited)
        self.first_frame_max_ratio: float = getattr(config, "first_frame_max_ratio", 0.0)

    def _validate_vision_token_ids(self, llm_path: str) -> None:
        """Confirm the vision token ids in config match the tokenizer vocab."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
        except Exception as e:  # Only warn when the tokenizer cannot be loaded; do not block
            print(f"[LLM][WARN] cannot load tokenizer to validate vision token ids: {e}")
            return

        expected = {
            "<|image_pad|>": self.image_pad_token_id,
            "<|vision_start|>": self.vision_start_token_id,
            "<|vision_end|>": self.vision_end_token_id,
        }
        unk_id = getattr(tokenizer, "unk_token_id", None)
        mismatches = []
        for token, cfg_id in expected.items():
            actual_id = tokenizer.convert_tokens_to_ids(token)
            if actual_id is None or actual_id == unk_id:
                mismatches.append(f"  {token}: not in the tokenizer vocab (config={cfg_id})")
            elif actual_id != cfg_id:
                mismatches.append(f"  {token}: config={cfg_id} but tokenizer={actual_id}")
        if mismatches:
            raise ValueError(
                "Vision token ids in LLMConfig do not match the tokenizer; continuing "
                "would write vision features to wrong token positions. Fix "
                "image_pad_token_id / vision_start_token_id / vision_end_token_id in the YAML:\n"
                + "\n".join(mismatches)
            )

    def _arrange_tokens(
        self,
        query_tokens: torch.Tensor,    # (B, S, Q, D)
        selected_vit: Optional[torch.Tensor],  # (B, S, K, D) or None
        strategy: str,
        arrangement: str,
    ) -> torch.Tensor:
        """Concatenate query + selected_vit into (B, total_tokens, D)."""
        b, s, q, d = query_tokens.shape
        if strategy == "query_only" or selected_vit is None:
            # (B, S, Q, D) -> (B, S*Q, D)
            return query_tokens.reshape(b, s * q, d)
        # strategy in {vit_only, query_and_vit}
        k = selected_vit.shape[2]
        if strategy == "vit_only":
            return selected_vit.reshape(b, s * k, d)
        # query_and_vit: always interleaved: [q_seg0, v_seg0, q_seg1, v_seg1, ...]
        out_segs = []
        for i in range(s):
            out_segs.append(query_tokens[:, i, :, :])  # (B, Q, D)
            out_segs.append(selected_vit[:, i, :, :])  # (B, K, D)
        return torch.cat(out_segs, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, seq_len)
        attention_mask: torch.Tensor,    # (B, seq_len)
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # (B, 3, T, H, W) or (B, 3, H, W) or None
        modality: str = "image",          # "image" | "video" | "text"
        token_plan: Optional[dict] = None,  # TokenPlan (passed through from the collator)
        visidx: Optional[torch.Tensor] = None,  # (B, L) video patch indices from visidx.npy
        uniform_sample_frames: bool = False,    # Uniform-sampling path (aligned with pretraining, video only)
        uniform_sample_n: int = 32,
        uniform_segment_t_size: int = 8,
        frame_cat_prob: float = 0.0,            # Frame-concatenation augmentation probability (uniform path during training only)
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1) Text-only
        if pixel_values is None or modality == "text":
            # Text-only skips the ViT / token_selector, so the returned loss has no sparsity
            # term. The cached values must be cleared explicitly; otherwise the training side
            # reads stale sparsity from the previous visual batch and subtracts it incorrectly
            # when splitting CE.
            self._last_sparsity_loss = None
            self._last_sparsity_loss_tensor = None
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # 2) ViT forward -> (query_tokens, vit_tokens)
        # Ensure pixel_values dtype matches the ViT weights
        vit_dtype = next(self.vit.parameters()).dtype
        query_tokens, vit_tokens, _ = self.vit(
            pixel_values=pixel_values.to(vit_dtype),
            visidx=visidx,
            modality=modality,
            uniform_sample_frames=uniform_sample_frames,
            uniform_sample_n=uniform_sample_n,
            uniform_segment_t_size=uniform_segment_t_size,
            frame_cat_prob=frame_cat_prob,
        )
        # query_tokens: (B, S, Q, D); vit_tokens: (B, S, P, D)

        # 3) Decide the composition according to the plan
        if token_plan is None:
            strategy = "query_and_vit"
            arrangement = "interleave"
        elif isinstance(token_plan, dict):
            strategy = token_plan["strategy"]
            arrangement = token_plan["arrangement"]
        else:
            # TokenPlan dataclass
            strategy = token_plan.strategy
            arrangement = token_plan.arrangement

        sparsity_loss = None
        selector_score_mean = None
        selector_score_max = None
        selector_score_min = None
        # When the token_selector is frozen, no DDP dummy forward / sparsity_loss is needed,
        # and inference-mode selection is used.
        selector_frozen = not any(p.requires_grad for p in self.token_selector.parameters())
        if strategy == "vit_only":
            # vit_only: skip the token selector and use all vit tokens directly
            selected_vit = vit_tokens  # (B, S, P, D)
            # DDP requires every parameter to be in the computation graph
            # (find_unused_parameters=False). The dummy forward that keeps gradient
            # connectivity is needed only when the token_selector is trainable.
            if self.training and not selector_frozen:
                _dummy = self.token_selector(query_tokens, vit_tokens, top_k=1)
                sparsity_loss = _dummy["selected_tokens"].sum() * 0.0
        elif strategy == "query_and_vit":
            # Compute top-K dynamically from vit_ratio and the actual patches per segment P.
            # vit_ratio must come from token_plan: the collator expanded the <|image_pad|>
            # placeholders using the plan-sampled ratio (K = round(P * plan.vit_ratio), see
            # data/collator.py::_expected_pad_count). Hardcoding a constant here would break
            # the vision-token / pad-count match whenever another ratio is sampled.
            P = vit_tokens.shape[2]
            vit_ratio = 1.0
            if isinstance(token_plan, dict):
                vit_ratio = float(token_plan.get("vit_ratio", 1.0))
            elif token_plan is not None:
                vit_ratio = float(getattr(token_plan, "vit_ratio", 1.0))
            K = max(1, round(P * vit_ratio))
            budget = float(round(P * 0.4))
            # First-frame token cap for video data (active during inference or when the
            # token_selector is frozen; the selector decides internally whether to apply
            # capping based on its own frozen state)
            ppf = None  # patches_per_frame
            ratio = 0.0
            if modality == "video" and self.first_frame_max_ratio > 0.0:
                image_size = self.config.vit.image_size
                patch_size = self.config.vit.patch_size
                ppf = (image_size // patch_size) ** 2
                ratio = self.first_frame_max_ratio
            sel_out = self.token_selector(
                query_tokens, vit_tokens, top_k=K,
                patches_per_frame=ppf,
                first_frame_max_ratio=ratio,
            )
            selected_vit = sel_out["selected_tokens"]  # (B, S, K, D)
            scores_detach = sel_out["scores"].detach()
            selector_score_mean = scores_detach.mean()
            selector_score_max = scores_detach.max()
            selector_score_min = scores_detach.min()
            # Sparsity constraint (active only when the token_selector is trainable):
            #   - Penalizes the total vit tokens retained per segment, pushing sum(scores)
            #     toward K (the budget).
            #   - Uses scores.sum(-1).mean() rather than scores.mean(); the latter pushes
            #     all sigmoid scores toward 0 (equivalent to preventing any token from
            #     being selected), while the former only constrains the "total selected
            #     token volume" and does not force every score toward zero.
            #   - OCR samples (vit_ratio=1.0) skip the sparsity loss: the full token set
            #     needs no selection constraint, and the meaningless gradients would
            #     disturb the token_selector.
            if self.training and not selector_frozen and self.sparsity_lambda > 0.0 and vit_ratio < 1.0:
                scores = sel_out["scores"]          # (B, S, P), carries gradients; values in [0, 1]
                # Goal: the per-segment sum of scores approximately equals K (token budget).
                # Use a two-sided L1 constraint: the larger the deviation from K (either
                # direction), the larger the penalty. This prevents scores from collapsing
                # to 0 overall (a one-sided hinge only penalizes exceeding, not collapse).
                # budget = float(K)
                sparsity_loss = self.sparsity_lambda * (scores.sum(dim=-1).mean() - budget).abs()
        else:
            selected_vit = None
            # query_only: the token_selector is not involved either; add a zero contribution forward
            if self.training and not selector_frozen:
                _dummy = self.token_selector(query_tokens, vit_tokens, top_k=1)
                sparsity_loss = _dummy["selected_tokens"].sum() * 0.0

        # 4) Concatenate
        all_vision = self._arrange_tokens(
            query_tokens=query_tokens,
            selected_vit=selected_vit,
            strategy=strategy,
            arrangement=arrangement,
        )  # (B, total_tokens, D_vit)

        # 5) Projector
        vision_embeds = self.projector(all_vision.to(self.language_model.dtype))  # (B, total_tokens, D_llm)

        # 6) masked_scatter into the vision pad positions
        emb_layer = self.language_model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids).clone()  # (B, seq_len, D_llm)

        n_vision = vision_embeds.shape[1]
        # Find all vision pad positions and fill them in order of appearance
        pad_mask = (input_ids == self.image_pad_token_id)  # (B, seq_len)
        n_pad_per_sample = pad_mask.sum(dim=1)  # (B,)

        if not torch.all(n_pad_per_sample == n_vision):
            # No silent truncation / error
            mismatch = (n_pad_per_sample != n_vision).nonzero(as_tuple=True)[0]
            raise ValueError(
                f"vision tokens ({n_vision}) do not match the <|image_pad|> count in input_ids "
                f"({n_pad_per_sample.tolist()}); mismatched sample indices: {mismatch.tolist()}"
            )

        inputs_embeds[pad_mask] = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        # 7) LLM forward
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,  # Disable the KV cache during training to save memory; inference goes through generate(), not this path
        )

        # 8) Add the sparsity regularization / DDP dummy gradient connectivity
        # sparsity_loss serves two purposes:
        #   a) The sparsity regularizer under the query_and_vit strategy (nonzero value);
        #   b) Keeping token_selector parameters in the computation graph under the
        #      query_only / vit_only strategies (value = 0, but the gradient path is kept
        #      so DDP with find_unused_parameters=False does not error).
        # When lm_out.loss is None (fully masked samples), add the dummy contribution to
        # the logits so the token_selector parameters are connected to the output tensor
        # whether or not a loss exists.
        if sparsity_loss is not None:
            if lm_out.loss is not None:
                new_loss = lm_out.loss + sparsity_loss
                new_logits = lm_out.logits
            else:
                # When loss is None, keep gradient connectivity through the logits.
                # Use * 0.0 rather than a direct add to guarantee the logits are numerically
                # unaffected (sparsity_loss may be nonzero).
                new_loss = None
                new_logits = lm_out.logits + sparsity_loss.sum() * 0.0
            lm_out = lm_out.__class__(
                loss=new_loss,
                logits=new_logits,
                past_key_values=lm_out.past_key_values,
                hidden_states=lm_out.hidden_states,
                attentions=lm_out.attentions,
            )

        # Cache selector stats on the module so the train loop can read them via _unwrap(model) after DDP wrapping
        self._last_selector_score_mean = selector_score_mean
        self._last_selector_score_max = selector_score_max
        self._last_selector_score_min = selector_score_min
        # Cache sparsity separately too: the returned lm_out.loss = CE + sparsity, and the
        # training side records them apart (CE needs token-count weighting; sparsity is a
        # per-batch regularizer and must not be mixed in).
        self._last_sparsity_loss_tensor = sparsity_loss
        self._last_sparsity_loss = (
            float(sparsity_loss.detach()) if sparsity_loss is not None else None
        )
        return lm_out

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------
    def freeze_for_stage(self, stage: str, trainable_modules: Optional[List[str]] = None) -> None:
        """
        stage: "alignment" | "mid_training" | "sft"
        trainable_modules: fine-grained control over ["vit", "projector", "token_selector", "llm"]
          - Each name is controlled independently, with no linkage
          - "projector"  -> train only the projector, not the token_selector
          - "token_selector" -> train only the token_selector, not the projector
          - To train both: ["projector", "token_selector"]
        """
        if trainable_modules is None:
            if stage == "alignment":
                trainable_modules = ["projector"]
            else:
                trainable_modules = ["projector", "token_selector", "llm"]

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        if "vit" in trainable_modules:
            for p in self.vit.parameters():
                p.requires_grad = True
        if "projector" in trainable_modules:
            for p in self.projector.parameters():
                p.requires_grad = True
        if "token_selector" in trainable_modules:
            for p in self.token_selector.parameters():
                p.requires_grad = True
        if "llm" in trainable_modules:
            for p in self.language_model.parameters():
                p.requires_grad = True

    def print_trainable(self) -> None:
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
