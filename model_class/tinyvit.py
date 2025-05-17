from functools import partial
from typing import Optional
from timm.layers import LayerNorm2d, NormMlpClassifierHead, trunc_normal_
from timm.models import checkpoint_seq
from timm.models.tiny_vit import PatchEmbed, ConvLayer, TinyVitStage, PatchMerging
from torch import nn
import torch


class TinyVit(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dims=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_sizes=(7, 7, 14, 7),
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.1,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            act_layer=nn.GELU,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio
        self.grad_checkpointing = use_checkpoint

        self.patch_embed = PatchEmbed(
            in_chs=in_chans,
            out_chs=embed_dims[0],
            act_layer=act_layer,
        )

        # stochastic depth rate rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.Sequential()
        stride = self.patch_embed.stride
        prev_dim = embed_dims[0]
        self.feature_info = []
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage = ConvLayer(
                    dim=prev_dim,
                    depth=depths[stage_idx],
                    act_layer=act_layer,
                    drop_path=dpr[:depths[stage_idx]],
                    conv_expand_ratio=mbconv_expand_ratio,
                )
            else:
                out_dim = embed_dims[stage_idx]
                drop_path_rate = dpr[sum(depths[:stage_idx]):sum(depths[:stage_idx + 1])]
                stage = TinyVitStage(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    drop_path=drop_path_rate,
                    downsample=PatchMerging,
                    act_layer=act_layer,
                )
                prev_dim = out_dim
                stride *= 2
            self.stages.append(stage)
            self.feature_info += [dict(num_chs=prev_dim, reduction=stride, module=f'stages.{stage_idx}')]

        # Classifier head
        self.num_features = embed_dims[-1]

        norm_layer_cf = partial(LayerNorm2d, eps=1e-5)
        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            norm_layer=norm_layer_cf,
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def TinyVit_11M(num_classes=10):
    return TinyVit(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1
    )


def TinyVit_21M(num_classes=10):
    return TinyVit(
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2
    )


