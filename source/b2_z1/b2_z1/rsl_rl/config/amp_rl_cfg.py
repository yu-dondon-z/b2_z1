from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Literal


from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg



#########################
# Policy configurations #
#########################
@configclass
class AMP_RslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    
    class_name: str = "AMP_ActorCritic"    



@configclass
class AMP_RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticRecurrentCfg):
    pass




############################
# Algorithm configurations #
############################


@configclass
class AMP_RslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    
    
    class_name: str = "AMP_PPO"
    
    
    
@configclass
class AMP_AlgorithmCfg:  
    
    amp_motion_files: str = MISSING
    
    amp_reward_coef: float = 0.2
    
    amp_discr_hidden_dims: list[int] = [1024, 512]

    amp_task_reward_lerp: float = 0.8
    
    amp_num_preload_transitions: int = 2000000
    
    min_normalized_std: list[float] = [0.01, 0.01, 0.01] * 4
    
    


#########################
# Runner configurations #
#########################

@configclass
class AMP_RslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    pass
