from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from envs.init_placement_env.CatanPerActionExtractor import CatanPerActionExtractor


class CatanPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CatanPerActionExtractor,
            features_extractor_kwargs=dict(hidden_dim=128),
        )
