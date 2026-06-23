# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

from isaaclab.utils import configclass

from b2_z1.assets import unitree_actuators


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
ROBOT_ASSETS_DIR = CUR_DIR + "/robots"
B2Z1_ASSETS_DIR = CUR_DIR + "/b2_z1"




@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        # enabled_self_collisions=False,  
        
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


""" Configuration for the Unitree robots."""


B2_Z1_GUN_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{B2Z1_ASSETS_DIR}/b2_z1_gun.urdf",
        self_collision=False,
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            
            # z1
            "joint1": 0.0,
            "joint2": 0.4, 
            "joint3": -0.4,
            "joint4": -0.4,
            "joint5": 0.0,
            "joint6": 0.0,
        },
        
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=200,
            velocity_limit_sim=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "2": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=320,
            velocity_limit_sim=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        
        # "z1_motor1": ImplicitActuatorCfg(
        #     joint_names_expr=["joint1", "joint3", "joint4", "joint5", "joint6"],
        #     # stiffness=60.0,
        #     # damping=1.0,
            
        #     friction=0.01,
        #     # effort_limit_sim=30.0,
        #     effort_limit_sim=90.0,
        #     velocity_limit_sim=3.1415,
            
        # ),
        
        # "z1_motor2": ImplicitActuatorCfg(
        #     joint_names_expr=["joint2"],
        #     # stiffness=60.0,
        #     # damping=1.0,
            
        #     friction=0.01,
        #     # effort_limit_sim=60.0,
        #     effort_limit_sim=90.0,
        #     velocity_limit_sim=3.1415,    
        # ),
        
        
        "z1_motor1": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", ],
            stiffness=250.0,
            damping=15.0,
            
            friction=0.01,
            # effort_limit_sim=30.0,
            effort_limit_sim=160.0,
            velocity_limit_sim=3.1415,
            
        ),
        
        "z1_motor2": ImplicitActuatorCfg(
            joint_names_expr=["joint4", "joint5", "joint6", ],
            stiffness=120.0,
            damping=5.0,
            
            friction=0.01,
            # effort_limit_sim=60.0,
            effort_limit_sim=120.0,
            velocity_limit_sim=3.1415,    
        ),
        
        
    },
)


