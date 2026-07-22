from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Literal


from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg



#########################
# Policy configurations #
#########################
@configclass
class HIM_RslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    
    class_name: str = "HIM_ActorCritic"    



@configclass
class HIM_RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticRecurrentCfg):
    pass




############################
# Algorithm configurations #
############################


@configclass
class HIM_RslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    
    
    class_name: str = "HIM_PPO"




#########################
# Runner configurations #
#########################

@configclass
class HIM_RslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the RSL-RL on-policy runner with HIM support."""
    
    history_length: int = 5
    """The length of the history to be used for the recurrent policy. Default is 0 (no history)."""

    privileged_history_length: int = 0
    """The length of the privileged history to be used for the recurrent policy. Default is 0 (no history)."""

