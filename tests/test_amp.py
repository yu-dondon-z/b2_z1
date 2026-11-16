"""CPU regressions for motion interpolation, AMP training, checkpoints, and resets."""

import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/b2_z1/b2_z1"

# Avoid launching Isaac Sim through b2_z1's task/UI registration imports.
package = types.ModuleType("b2_z1")
package.__path__ = [str(PACKAGE)]
with patch.dict(sys.modules, {"b2_z1": package}):
    from b2_z1.amp_config import AMP_JOINT_NAMES, MOTION_FILES
    from b2_z1.rsl_rl.datasets import AMPLoader
    from b2_z1.rsl_rl.networks import AMP_Discriminator
    from b2_z1.rsl_rl.storage import ReplayBuffer
    from b2_z1.rsl_rl.utils.amp_utils import quaternion_slerp
    from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner
    from b2_z1.rsl_rl.utils.vd5_curriculum_state import capture_vd5_curriculum_state


class MotionTests(unittest.TestCase):
    def loader(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return AMPLoader(device="cpu", time_between_frames=0.02, motion_files=MOTION_FILES, **kwargs)

    def test_all_reference_motions_and_transition_dimensions(self):
        loader = self.loader()
        self.assertEqual(loader.num_motions, 19)
        self.assertEqual(loader.observation_dim, 43)
        for states, next_states in loader.feed_forward_generator(2, 32):
            self.assertEqual(states.shape, (32, 43))
            self.assertTrue(torch.isfinite(states).all())
            self.assertTrue(torch.isfinite(next_states).all())

    def test_interpolation_endpoints_midpoint_and_preloaded_transitions(self):
        loader = self.loader(preload_transitions=True, num_preload_transitions=64)
        for index, trajectory in enumerate(loader.trajectories_full):
            duration = loader.trajectory_lens[index]
            frames = loader.get_full_frame_at_time_batch(np.array([index, index]), np.array([0.0, duration]))
            torch.testing.assert_close(frames[0], trajectory[0])
            torch.testing.assert_close(frames[1], trajectory[-1])
            middle = loader.get_full_frame_at_time(index, loader.trajectory_frame_durations[index] / 2)
            torch.testing.assert_close(middle[7:], (trajectory[0, 7:] + trajectory[1, 7:]) / 2)
        states, following = next(loader.feed_forward_generator(1, 16))
        self.assertEqual(states.shape, (16, 43))
        self.assertEqual(following.shape, (16, 43))

    def test_empty_motion_set_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "No AMP motion files"):
            AMPLoader(device="cpu", time_between_frames=0.02, motion_files=[])

    def test_quaternion_interpolation_is_normalized_and_does_not_mutate(self):
        first = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        second = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        first_copy, second_copy = first.clone(), second.clone()
        result = quaternion_slerp(first, second, torch.tensor([[0.5]]))
        torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]))
        torch.testing.assert_close(first, first_copy)
        torch.testing.assert_close(second, second_copy)
        torch.testing.assert_close(quaternion_slerp(first, -first, torch.tensor([[0.5]])), first)

    def test_one_environment_keeps_reward_batch_dimension(self):
        discriminator = AMP_Discriminator(86, 0.2, [16, 8], "cpu", task_reward_lerp=0.8)
        reward, _ = discriminator.predict_amp_reward(torch.zeros(1, 43), torch.zeros(1, 43), torch.ones(1))
        self.assertEqual(reward.shape, (1,))

    def test_replay_wrap_and_oversized_insert(self):
        buffer = ReplayBuffer(1, 4, "cpu")
        buffer.insert(torch.arange(3.0).unsqueeze(-1), torch.arange(3.0).unsqueeze(-1) + 1)
        buffer.insert(torch.arange(3.0, 6.0).unsqueeze(-1), torch.arange(4.0, 7.0).unsqueeze(-1))
        self.assertEqual(buffer.num_samples, 4)
        self.assertEqual(set(buffer.states[:, 0].tolist()), {2.0, 3.0, 4.0, 5.0})
        buffer.insert(torch.arange(10.0).unsqueeze(-1), torch.arange(10.0).unsqueeze(-1) + 1)
        torch.testing.assert_close(buffer.states[:, 0], torch.arange(6.0, 10.0))


class FakeEnv:
    num_envs, num_actions, max_episode_length, step_dt = 4, 18, 100, 0.02
    device = "cpu"

    def __init__(self):
        self.env = self.unwrapped = self
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        limits = torch.tensor([-2.0, 2.0]).expand(1, 18, 2)
        self.scene = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(joint_pos_limits=limits))}

    def get_observations(self):
        return torch.zeros(4, 10), {"observations": {"critic": torch.zeros(4, 12), "amp_obs": torch.zeros(4, 43)}}

    def get_amp_observations(self):
        return torch.randn(4, 43)

    def step(self, actions):
        observations, extras = self.get_observations()
        extras.update(reset_env_ids=torch.tensor([1]), terminal_amp_states=torch.full((1, 43), 7.0))
        return observations, torch.ones(4), torch.tensor([0, 1, 0, 0]), extras


def load_vd3_symmetry():
    """Import the torch-only mirror implementation without task registration."""
    mdp_path = PACKAGE / "tasks/manager_based/b2_z1/mdp"
    package_name = "vd3_amp_symmetry_cpu"
    symmetry_package = types.ModuleType(package_name)
    symmetry_package.__path__ = [str(mdp_path)]
    with patch.dict(sys.modules, {package_name: symmetry_package}):
        spec = importlib.util.spec_from_file_location(
            package_name + ".vd3_symmetry", mdp_path / "vd3_symmetry.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeSymmetryEnv(FakeEnv):
    """VD3 actor layout and articulation metadata with the real interleaving."""

    def __init__(self, symmetry):
        super().__init__()
        robot = self.scene["robot"]
        robot.joint_names, robot.device = symmetry.JOINT_NAMES, "cpu"
        robot.data.default_joint_pos = torch.tensor([
            0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, 0.0,
            -1.5, -1.5, -1.5, -1.5, 2.55, -1.6, -1.4, 0.0, 0.0,
        ]).expand(self.num_envs, -1)
        robot.data.default_joint_vel = torch.zeros(self.num_envs, 18)
        self.observation_manager = types.SimpleNamespace(
            active_terms={"policy": symmetry._POLICY_TERMS},
            group_obs_term_dim={"policy": symmetry._POLICY_DIMS},
        )
        action = types.SimpleNamespace(
            _joint_names=robot.joint_names, _scale=0.25,
            cfg=types.SimpleNamespace(use_default_offset=True),
        )
        self.action_manager = types.SimpleNamespace(
            active_terms=["JointPositionAction"], get_term=lambda name: action,
        )

    def get_observations(self):
        observations = torch.linspace(-0.3, 0.3, 70).repeat(self.num_envs, 1)
        observations[:, 3:6] = torch.tensor([-1.02, 0.03, -0.3])
        observations[:, 6:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return observations, {"observations": {
            "critic": torch.zeros(self.num_envs, 12),
            "amp_obs": torch.zeros(self.num_envs, 43),
        }}


def runner_config():
    return {
        "algorithm": {"class_name": "AMP_PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
        "policy": {"class_name": "AMP_ActorCritic", "actor_hidden_dims": [16], "critic_hidden_dims": [16]},
        "amp_cfg": {
            "amp_motion_files": MOTION_FILES, "amp_reward_coef": 0.2,
            "amp_num_preload_transitions": 64, "amp_discr_hidden_dims": [16, 8],
            "amp_task_reward_lerp": 0.8, "min_normalized_std": [0.01] * 18,
        },
        "num_steps_per_env": 2, "save_interval": 10, "empirical_normalization": False,
    }


class TrainingTests(unittest.TestCase):
    def test_real_amp_update_applies_mirror_loss_to_legs_without_augmenting_ppo(self):
        torch.manual_seed(23)
        symmetry = load_vd3_symmetry()
        cfg = runner_config()
        leg_indices = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)
        arm_indices = (8, 13, 14, 15, 16, 17)
        cfg["algorithm"]["symmetry_cfg"] = {
            "use_data_augmentation": False, "use_mirror_loss": True,
            "mirror_loss_coeff": 0.2, "mirror_action_indices": leg_indices,
            "data_augmentation_func": symmetry.vd3_left_right,
        }
        env = FakeSymmetryEnv(symmetry)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = AMP_OnPolicyRunner(env, cfg, device="cpu")
        # An asymmetric, deterministic output makes a missing mirror term fail
        # independently of a lucky random initialization.
        with torch.no_grad():
            runner.alg.policy.actor[-1].weight.zero_()
            runner.alg.policy.actor[-1].bias.copy_(torch.arange(1.0, 19.0) / 10)

        actor_outputs, critic_batches, likelihood_batches, updates = [], [], [], []

        def retain_actor_output(module, args, output):
            if torch.is_grad_enabled():
                output.retain_grad()
                actor_outputs.append(output)

        def record_critic_batch(module, args, output):
            if torch.is_grad_enabled():
                critic_batches.append(args[0].shape[0])

        original_log_prob = runner.alg.policy.get_actions_log_prob

        def record_likelihood(actions):
            if torch.is_grad_enabled():
                likelihood_batches.append(tuple(actions.shape))
            return original_log_prob(actions)

        original_update = runner.alg.update

        def record_update():
            self.assertEqual(tuple(runner.alg.storage.observations.shape), (2, 4, 70))
            self.assertEqual(tuple(runner.alg.storage.actions.shape), (2, 4, 18))
            updates.append(original_update())
            return updates[-1]

        actor_hook = runner.alg.policy.actor.register_forward_hook(retain_actor_output)
        critic_hook = runner.alg.policy.critic.register_forward_hook(record_critic_batch)
        try:
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                runner.alg.policy, "get_actions_log_prob", side_effect=record_likelihood
            ), patch.object(runner.alg, "update", side_effect=record_update):
                runner.learn(1)
        finally:
            actor_hook.remove()
            critic_hook.remove()

        self.assertEqual(len(updates), 1)
        self.assertTrue(all(math.isfinite(value) for value in updates[0].values()))
        self.assertGreater(updates[0]["symmetry"], 0)
        self.assertEqual([tuple(output.shape) for output in actor_outputs], [(8, 18), (16, 18)])
        self.assertEqual(critic_batches, [8])
        self.assertEqual(likelihood_batches, [(8, 18)])
        self.assertTrue(env._vd3_leg_symmetry_validated)

        # This second actor invocation feeds only the mirror branch. Its output
        # gradient therefore isolates the new loss from PPO/entropy gradients.
        mirror_grad = actor_outputs[1].grad
        self.assertIsNotNone(mirror_grad)
        self.assertTrue(torch.isfinite(mirror_grad).all())
        torch.testing.assert_close(mirror_grad[:8], torch.zeros_like(mirror_grad[:8]), atol=0, rtol=0)
        torch.testing.assert_close(
            mirror_grad[8:, list(arm_indices)], torch.zeros(8, 6), atol=0, rtol=0
        )
        self.assertTrue((mirror_grad[8:, list(leg_indices)].abs() > 0).all())

    def test_amp_constructor_rejects_invalid_mirror_action_masks(self):
        symmetry = load_vd3_symmetry()
        for indices in ([], [0, 0], [-1], [18], [1.0], [True]):
            with self.subTest(indices=indices):
                cfg = runner_config()
                cfg["algorithm"]["symmetry_cfg"] = {
                    "use_data_augmentation": False, "use_mirror_loss": True,
                    "mirror_loss_coeff": 0.2, "mirror_action_indices": indices,
                    "data_augmentation_func": symmetry.vd3_left_right,
                }
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                    ValueError, "mirror_action_indices"
                ):
                    AMP_OnPolicyRunner(FakeSymmetryEnv(symmetry), cfg, device="cpu")

    def test_training_uses_terminal_states_and_checkpoint_restores_prior(self):
        cfg = runner_config()
        with contextlib.redirect_stdout(io.StringIO()):
            runner = AMP_OnPolicyRunner(FakeEnv(), cfg, device="cpu")
            before = [p.detach().clone() for p in runner.alg.discriminator.parameters()]
            runner.learn(1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, runner.alg.discriminator.parameters())))
        self.assertEqual(cfg["algorithm"]["class_name"], "AMP_PPO")
        self.assertEqual(cfg["policy"]["class_name"], "AMP_ActorCritic")
        # Every second transition is a terminated env. Its next state is the saved
        # physical terminal state, not a new episode's randomly reset state.
        torch.testing.assert_close(runner.alg.amp_storage.next_states[1], torch.full((43,), 7.0))
        self.assertGreater(runner.alg.amp_normalizer.count, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model.pt")
            runner.save(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                restored = AMP_OnPolicyRunner(FakeEnv(), cfg, device="cpu")
            restored.load(checkpoint)
        for original, loaded in zip(runner.alg.discriminator.parameters(), restored.alg.discriminator.parameters()):
            torch.testing.assert_close(original, loaded)
        np.testing.assert_array_equal(runner.alg.amp_normalizer.mean, restored.alg.amp_normalizer.mean)
        np.testing.assert_array_equal(runner.alg.amp_normalizer.var, restored.alg.amp_normalizer.var)
        self.assertEqual(runner.alg.amp_normalizer.count, restored.alg.amp_normalizer.count)


class ResetTests(unittest.TestCase):
    def test_partial_reference_reset_uses_leg_ids_world_velocities_and_resamples_commands(self):
        from test_curve_tracking import quat_apply

        events = []

        class BaseEnv:
            def _reset_idx(self, env_ids):
                events.append("default_reset")

        stubs = {
            "b2_z1": package,
            "b2_z1.amp_config": types.SimpleNamespace(AMP_JOINT_NAMES=AMP_JOINT_NAMES),
            "b2_z1.rsl_rl.datasets": types.SimpleNamespace(AMPLoader=AMPLoader),
            "b2_z1.envs": types.ModuleType("b2_z1.envs"),
            "b2_z1.envs.b2z1_rl_env": types.SimpleNamespace(B2Z1ManagerBasedRLEnv=BaseEnv),
            "isaaclab.utils.math": types.SimpleNamespace(
                quat_apply=quat_apply, convert_quat=lambda q, to: torch.roll(q, 1, dims=-1)
            ),
        }
        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "b2_z1.envs.b2z1_rl_amp_env", PACKAGE / "envs/b2z1_rl_amp_env.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        env = object.__new__(module.B2Z1ManagerAMPBasedRLEnv)
        env.device = "cpu"
        env.cfg = types.SimpleNamespace(
            reference_state_initialization_prob=0.5,
            commands=types.SimpleNamespace(base_velocity=types.SimpleNamespace(ground_height=0.1)),
        )
        env._pre_pre_action = torch.ones(3, 18)
        env.amp_joint_ids = [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13]
        frame = torch.zeros(1, 49)
        frame[:, :3] = torch.tensor([0.2, 0.3, 0.45])
        frame[:, 3:7] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
        frame[:, 7:19] = torch.arange(12.0)
        frame[:, 31:34] = torch.tensor([1.0, 0.0, 0.0])
        frame[:, 34:37] = torch.tensor([0.0, 1.0, 0.0])
        frame[:, 37:49] = torch.arange(12.0) * 2
        env.amp_loader = types.SimpleNamespace(get_full_frame_batch=lambda count: frame.repeat(count, 1))

        class Robot:
            def write_joint_state_to_sim(self, pos, vel, **kwargs):
                self.joints = pos, vel, kwargs

            def write_root_pose_to_sim(self, pose, **kwargs):
                self.pose = pose

            def write_root_velocity_to_sim(self, velocity, **kwargs):
                self.velocity = velocity

        class Scene(dict):
            env_origins = torch.tensor([[10.0, 20.0, 2.0], [0.0, 0.0, 0.0], [30.0, 40.0, 0.0]])

            def write_data_to_sim(self):
                events.append("write")

        robot = Robot()
        env.scene = Scene(robot=robot)
        env.sim = types.SimpleNamespace(forward=lambda: events.append("forward"))
        resampled = []
        env.command_manager = types.SimpleNamespace(reset=lambda ids: (events.append("commands"), resampled.append(ids)))
        with patch.object(torch, "rand", return_value=torch.tensor([0.2, 0.8])):
            env._reset_idx(torch.tensor([0, 2]))
        self.assertEqual(events, ["default_reset", "write", "forward", "commands"])
        self.assertEqual(robot.joints[2]["joint_ids"], env.amp_joint_ids)
        self.assertEqual(robot.joints[2]["env_ids"].tolist(), [0])
        torch.testing.assert_close(robot.joints[0], frame[:, 7:19])
        torch.testing.assert_close(robot.joints[1], frame[:, 37:49])
        torch.testing.assert_close(robot.pose[:, :3], torch.tensor([[10.2, 20.3, 2.55]]))
        torch.testing.assert_close(robot.velocity, torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)
        self.assertEqual(resampled[0].tolist(), [0])
        torch.testing.assert_close(env._pre_pre_action[0], torch.zeros(18))
        torch.testing.assert_close(env._pre_pre_action[1], torch.ones(18))


if __name__ == "__main__":
    unittest.main()
