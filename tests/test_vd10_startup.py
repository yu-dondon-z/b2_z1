"""CPU regressions for VD10's physical pipe clearance and reset handoff.

The real VD10/VD5 command inheritance is loaded above the existing CPU nominal
command fixture. Isaac Sim is never imported; the environment reset override
and configuration function are compiled directly from their source ASTs.
"""

import ast
import copy
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch

import test_vd5_commands as vd5_tests
import test_vd5_startup as startup_tests


MDP = vd5_tests.MDP
ROOT = vd5_tests.ROOT
tcp = sys.modules["vd5_cpu.tcp_frame"]
tube = vd5_tests.load("vd5_cpu.vd10_geometry", MDP / "vd10_geometry.py")


def euler_quaternion(roll, pitch, yaw):
    """Small CPU replacement for the simulator's wxyz Euler helper."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), dim=-1)


with patch.dict(sys.modules, {
    "isaaclab.utils.math": types.SimpleNamespace(quat_from_euler_xyz=euler_quaternion),
    "vd5_cpu.commands.vd5_startup_command": startup_tests.startup,
}):
    vd10 = vd5_tests.load("vd5_cpu.commands.vd10_safe_seam_command",
                         MDP / "commands/vd10_safe_seam_command.py")


def source_declaration(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def quaternion(angles):
    return euler_quaternion(*torch.tensor(angles, dtype=torch.float32).unbind(-1))


def command_fixture():
    fixture = startup_tests.StartupTests().command()
    command = vd10.VD10SafeSeamCommand.__new__(vd10.VD10SafeSeamCommand)
    command.__dict__.update(fixture.__dict__)
    command.cfg.vd5_ground_transition_steps = 20_000
    command.cfg.vd5_seam_trailing_distance = command.trailing_distance = .8
    command._vd5_ground_transition_steps = 20_000
    command._vd5_last_ground_curriculum_step = 0
    command.env.common_step_counter = 0
    command.nominal = torch.tensor([[-.95, 0., .35]]).repeat(2, 1)
    command.pose_target[:, :3] = command.nominal
    command.robot.data.body_pos_w[:, 0, 2] = .5
    command.robot.data.body_state_w[:, 0, 2] = .5
    command.physical_pos = torch.tensor([[-.95, 0., .85], [1.05, 0., .85]])
    command.physical_quat = quaternion([[.1, .25, 2.9], [-.1, .35, -2.9]])
    return command


def physical_tcp(command):
    return patch.object(startup_tests.startup, "tcp_pose_w",
                        side_effect=lambda _: (command.physical_pos, command.physical_quat))


def target_axis(command):
    return tcp.quat_apply(command._get_target_quat_w(), torch.tensor([[1., 0., 0.]]).repeat(2, 1))


class VD10CommandTests(unittest.TestCase):
    def test_startup_preserves_measured_pose_and_continuous_shortest_quaternion(self):
        command = command_fixture()
        command._vd5_ground_blend = .6
        # A negative quaternion representation must not cause a sign jump.
        command.physical_quat *= -1
        with physical_tcp(command):
            command._update_command()
            torch.testing.assert_close(command._get_target_pos_w(), command.physical_pos)
            first = command._get_target_quat_w().clone()
            torch.testing.assert_close(first, command.physical_quat, atol=2e-6, rtol=0)
            previous = first
            for steps in (1, 25, 50, 75, 99, 100, 101):
                command.env.episode_length_buf[:] = steps
                command._update_command()
                current = command._get_target_quat_w().clone()
                torch.testing.assert_close(current.norm(dim=-1), torch.ones(2))
                self.assertTrue(((previous * current).sum(-1) > 0).all())
                if steps == 50:
                    nominal_w = command.robot.data.body_pos_w[:, 0] + command.nominal
                    seam_w = torch.tensor([[-.8, 0., 0.], [1.2, 0., 0.]])
                    effective = .4 * nominal_w + .6 * seam_w
                    expected = .5 * command.physical_pos + .5 * effective
                    axis = target_axis(command)
                    height = (1.1 * axis[:, 2]).clamp_min(0.) + .028 * axis[:, :2].norm(dim=-1) + .005
                    expected[:, 2] += .5 * .6 * height
                    torch.testing.assert_close(command._get_target_pos_w(), expected, atol=2e-6, rtol=0)
                if steps in (1, 100, 101):
                    torch.testing.assert_close(current, previous, atol=3e-5, rtol=0)
                previous = current
        desired = tcp.quat_mul(command.robot.data.body_state_w[:, 0, 3:7],
                               command.pose_target[:, 3:])
        torch.testing.assert_close((desired * previous).sum(-1).abs(), torch.ones(2))

    def test_pitch_blend_seam_xy_and_five_mm_surface_with_rotated_base(self):
        command = command_fixture()
        base_quat = quaternion([[0., 0., 0.], [.17, -.22, .8]])
        command.robot.data.body_state_w[:, 0, 3:7] = base_quat
        command._startup_initialized[:] = True
        command._startup_pos_w[:] = command.physical_pos
        command._startup_quat_w[:] = command.physical_quat
        command.env.episode_length_buf[:] = 100
        base_pos = command.robot.data.body_pos_w[:, 0]
        nominal_w = base_pos + tcp.quat_apply(base_quat, command.nominal)
        seam_w = torch.tensor([[-.8, 0., 0.], [1.2, 0., 0.]])
        for blend in (0., .5, 1.):
            with self.subTest(ground_blend=blend):
                command._vd5_ground_blend = blend
                command._update_command()
                target_w = command._get_target_pos_w()
                torch.testing.assert_close(target_w[:, :2],
                    ((1 - blend) * nominal_w + blend * seam_w)[:, :2], atol=2e-6, rtol=0)
                pitch = vd10.SAFE_PITCH_RAD + blend * (vd10.SEAM_PITCH_RAD - vd10.SAFE_PITCH_RAD)
                axis_b = tcp.quat_apply(command.pose_target[:, 3:],
                                       torch.tensor([[1., 0., 0.]]).repeat(2, 1))
                torch.testing.assert_close(axis_b,
                    torch.tensor([[-math.cos(pitch), 0., -math.sin(pitch)]]).repeat(2, 1),
                    atol=2e-6, rtol=0)
                if blend == 1.:
                    axis = target_axis(command)
                    # Independently construct the lowest front and rear rim points.
                    up = torch.tensor([[0., 0., 1.]]).repeat(2, 1)
                    rim_up = up - axis * axis[:, 2:3]
                    rim_up /= rim_up.norm(dim=-1, keepdim=True)
                    front_low = target_w - .028 * rim_up
                    rear_low = front_low - 1.1 * axis
                    minimum_z = torch.minimum(front_low[:, 2], rear_low[:, 2])
                    torch.testing.assert_close(minimum_z, torch.full((2,), .005), atol=2e-6, rtol=0)
                    torch.testing.assert_close(tube.cylinder_surface_min_z(target_w, axis), minimum_z)
                torch.testing.assert_close(command.pose_target_pos_w, target_w)
                torch.testing.assert_close(base_pos + tcp.quat_apply(base_quat, command.pose_command[:, :3]), target_w)

    def test_repeated_update_does_not_accumulate_height_or_restart_startup(self):
        command = command_fixture()
        command._vd5_ground_blend = .43
        with physical_tcp(command):
            command._update_command()
            start = command._startup_pos_w.clone()
            command.env.episode_length_buf[:] = 50
            command._update_command()
            before = command.pose_command.clone()
            world_before = command._get_target_pos_w().clone()
            for _ in range(3):
                command._update_command()
                torch.testing.assert_close(command.pose_command, before)
                torch.testing.assert_close(command._get_target_pos_w(), world_before)
                torch.testing.assert_close(command._startup_pos_w, start)
                torch.testing.assert_close(command.metrics["startup_fraction"], torch.full((2,), .5))


class VD10CylinderTests(unittest.TestCase):
    def test_finite_cylinder_includes_rear_end_and_tilted_rim(self):
        axes = torch.tensor([[0., 0., -1.], [0., 0., 1.], [1., 0., 0.], [.6, 0., .8]])
        height = tube.cylinder_tip_height_above_plane(axes, clearance=.005)
        torch.testing.assert_close(height, torch.tensor([.005, 1.105, .033, .9018]))
        tips = torch.zeros(4, 3)
        tips[:, 2] = height
        torch.testing.assert_close(tube.cylinder_surface_min_z(tips, axes),
                                   torch.full((4,), .005), atol=1e-7, rtol=0)
        seam_axis = torch.tensor([[-math.cos(vd10.SEAM_PITCH_RAD), 0., -math.sin(vd10.SEAM_PITCH_RAD)]])
        self.assertAlmostEqual(tube.cylinder_surface_min_z(torch.zeros(1, 3), seam_axis).item(),
                               -.012179034955114448, places=7)


class VD10ResetTests(unittest.TestCase):
    def test_repeated_partial_reset_captures_after_counter_zero_and_final_forward(self):
        class ParentReset:
            def _reset_idx(self, env_ids):
                # Match IsaacLab's ordering, including a stale physical cache.
                self.pending_pos[env_ids] = self.nominal_reset_pos[env_ids]
                self.command_manager.reset(env_ids)
                self.episode_length_buf[env_ids] = 0
                if self.reference_reset:
                    self.pending_pos[env_ids] = self.reference_reset_pos[env_ids]
                    self.pending_quat[env_ids] = self.reference_reset_quat[env_ids]
                    self.scene.write_data_to_sim()
                    self.sim.forward()
                    self.command_manager.reset(env_ids)

        reset_class = source_declaration(ROOT / "source/b2_z1/b2_z1/envs/b2z1_rl_amp_env.py",
            "B2Z1VD10ManagerAMPBasedRLEnv", {"B2Z1ManagerAMPBasedRLEnv": ParentReset})
        for reference in (False, True):
            with self.subTest(reference_reset=reference):
                command = command_fixture()
                command._vd5_ground_blend = 1.
                env = reset_class.__new__(reset_class)
                env.__dict__.update(command.env.__dict__)
                command.env = env
                events = []
                base = env.command_manager.get_term("base_velocity")

                def reset_commands(ids):
                    events.append(("command_reset", env.episode_length_buf[ids].clone()))
                    command.pose_command[ids, :3] = command.nominal[ids]
                    command._set_seam_target(ids)

                def forward():
                    events.append(("forward", None))
                    command.physical_pos[:] = env.pending_pos
                    command.physical_quat[:] = env.pending_quat

                env.command_manager = types.SimpleNamespace(get_term=lambda _: base, reset=reset_commands)
                env.scene = types.SimpleNamespace(write_data_to_sim=lambda: events.append(("write", None)))
                env.sim = types.SimpleNamespace(forward=forward)
                env.reference_reset = reference
                env.pending_pos = command.physical_pos.clone()
                env.pending_quat = command.physical_quat.clone()
                env.nominal_reset_pos = command.physical_pos + torch.tensor([[-.1, .1, .2]])
                env.reference_reset_pos = command.physical_pos + torch.tensor([[.1, -.1, .1]])
                env.reference_reset_quat = quaternion([[.15, .2, 2.8], [.2, -.1, -2.8]])
                ids = torch.tensor([0])
                with physical_tcp(command):
                    command._update_command()
                    for _ in range(2):
                        env.episode_length_buf[:] = 100
                        command._update_command()
                        untouched = command.pose_command[1].clone()
                        untouched_start = command._startup_pos_w[1].clone()
                        env._reset_idx(ids)
                        self.assertEqual([name for name, _ in events[-3:]],
                                         ["write", "forward", "command_reset"])
                        torch.testing.assert_close(events[-1][1], torch.zeros(1, dtype=torch.long))
                        torch.testing.assert_close(command._startup_pos_w[0], command.physical_pos[0])
                        torch.testing.assert_close(command._get_target_pos_w()[0], command.physical_pos[0])
                        torch.testing.assert_close(command._get_target_quat_w()[0],
                                                   command.physical_quat[0], atol=2e-6, rtol=0)
                        torch.testing.assert_close(command.pose_command[1], untouched)
                        torch.testing.assert_close(command._startup_pos_w[1], untouched_start)
                        self.assertEqual(env.episode_length_buf.tolist(), [0, 100])
                        # The automatic-reset path also computes commands before its next action.
                        command._update_command()
                        torch.testing.assert_close(command._get_target_pos_w()[0], command.physical_pos[0])
                        env.nominal_reset_pos[0, 2] += .05
                        env.reference_reset_pos[0, 2] += .05


class FakeAsset:
    def __init__(self):
        self.init_state = types.SimpleNamespace(pos=(0., 0., .5), joint_pos={})

    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result


configure_vd10 = source_declaration(MDP.parent / "b2_z1_vd10_env_cfg.py", "_configure_hollow_gun", {
    "B2_Z1_HOLLOW_GUN_CFG": FakeAsset(), "VD10SafeSeamCommand": vd10.VD10SafeSeamCommand,
    "SAFE_PITCH_RAD": vd10.SAFE_PITCH_RAD,
})


def curriculum_fixture(counter=0, play=False):
    env, base, old_arm = vd5_tests.curriculum_env(counter)
    command = command_fixture()
    command.cfg = old_arm.cfg
    for ranges in (command.cfg.ranges, command.cfg.limit_ranges):
        ranges.roll, ranges.yaw = (0., 0.), (math.pi, math.pi)
    command._last_pos_curriculum_step = 0
    command._vd5_last_ground_curriculum_step = counter
    command.env = env
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.step_dt = .02
    base.cfg.vd5_startup_duration_s = command.cfg.vd5_startup_duration_s = 2.
    base.curve_points_w = command_fixture().env.command_manager.get_term("base_velocity").curve_points_w
    env.command_manager = types.SimpleNamespace(get_term={"base_velocity": base, "ee_target": command}.__getitem__)
    env.cfg = types.SimpleNamespace(reference_state_initialization_prob=.8, vd5_nominal_base_height_m=.5,
        commands=types.SimpleNamespace(ee_target=command.cfg, base_velocity=base.cfg),
        scene=types.SimpleNamespace(robot=FakeAsset()),
        events=types.SimpleNamespace(reset_robot_joints=types.SimpleNamespace(params={"velocity_range": (0., 0.)})))
    configure_vd10(env.cfg, play=play)
    command._vd5_ground_transition_steps = command.cfg.vd5_ground_transition_steps
    command._vd5_ground_blend = 1. if play else 0.
    return env, command


class VD10CurriculumTests(unittest.TestCase):
    def test_profile_roundtrip_rebases_clock_and_advances_pitch(self):
        source, source_command = curriculum_fixture(123)
        source_command._vd5_ground_blend = .43
        saved = vd5_tests.state.capture_vd5_curriculum_state(source)
        self.assertEqual(saved["startup_profile"]["vd10"]["version"], 2)
        self.assertEqual(saved["startup_profile"]["vd10"]["arm_joint_reset_rad"],
                         [source.cfg.scene.robot.init_state.joint_pos[f"joint{i}"] for i in range(1, 7)])
        target, restored = curriculum_fixture(7)
        vd5_tests.state.restore_vd5_curriculum_state(target, saved)
        self.assertEqual(vd5_tests.state.capture_vd5_curriculum_state(target), saved)
        self.assertEqual(restored._vd5_last_ground_curriculum_step, 7)
        target.common_step_counter += 24
        with physical_tcp(restored):
            restored._update_command()
        self.assertAlmostEqual(restored._vd5_ground_blend, .43 + 24 / 20_000)
        pitch = vd10.SAFE_PITCH_RAD + restored._vd5_ground_blend * (vd10.SEAM_PITCH_RAD - vd10.SAFE_PITCH_RAD)
        expected = quaternion([[0., pitch, math.pi]]).repeat(2, 1)
        torch.testing.assert_close(restored.pose_target[:, 3:], expected)

    def test_old_or_changed_vd10_profile_is_rejected_without_mutation(self):
        env, _ = curriculum_fixture()
        before = vd5_tests.state.capture_vd5_curriculum_state(env)
        old = copy.deepcopy(before)
        old["startup_profile"].pop("vd10")
        changed = copy.deepcopy(before)
        changed["startup_profile"]["vd10"]["arm_joint_reset_rad"][1] += .01
        for incompatible in (old, changed):
            with self.subTest(profile=incompatible["startup_profile"]):
                with self.assertRaisesRegex(ValueError, "startup profile"):
                    vd5_tests.state.restore_vd5_curriculum_state(env, incompatible)
                self.assertEqual(vd5_tests.state.capture_vd5_curriculum_state(env), before)

    def test_train_and_play_use_safe_workspace_but_play_starts_at_full_task(self):
        train, train_command = curriculum_fixture()
        play, play_command = curriculum_fixture(play=True)
        self.assertEqual(train_command.cfg.vd5_ground_transition_steps, 20_000)
        self.assertEqual(play_command.cfg.vd5_ground_transition_steps, 0)
        self.assertEqual(play_command._vd5_ground_blend, 1.)
        for env, command in ((train, train_command), (play, play_command)):
            for axis in ("pos_x", "pos_y", "pos_z", "pitch"):
                self.assertEqual(getattr(command.cfg.ranges, axis), getattr(command.cfg.limit_ranges, axis))
            self.assertGreater(command.cfg.ranges.pos_z[0], 0.)
            self.assertEqual(command.cfg.ranges.pitch, (vd10.SAFE_PITCH_RAD, vd10.SAFE_PITCH_RAD))
            self.assertIs(command.cfg.class_type, vd10.VD10SafeSeamCommand)


if __name__ == "__main__":
    unittest.main()
