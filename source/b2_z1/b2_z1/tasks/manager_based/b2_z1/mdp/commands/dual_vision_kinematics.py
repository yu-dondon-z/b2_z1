"""Pure URDF encoder kinematics for propagating a captured visual TCP.

``position(q)`` maps measured encoder angles in an explicit joint-name order
to the URDF TCP origin in the requested base frame.  The defaults remain
the six-joint Z1 chain, ``dummy_target`` in ``base_link``.  It
includes every fixed mounting/tool offset on that chain and does not read a
simulator, a robot link pose, an actor, or an inverse-kinematics solution.
This model may supply a *delta* between two encoder samples; its output must
never be labelled as a camera measurement.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    return (identity + torch.sin(angle)[:, None, None] * skew
            + (1 - torch.cos(angle))[:, None, None] * (skew @ skew))


def _triple(text: str, field: str) -> list[float]:
    result = [float(value) for value in text.split()]
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid URDF {field}: expected three finite coordinates")
    return result


class URDFToolKinematics(nn.Module):
    joint_names = tuple(f"joint{i}" for i in range(1, 7))

    def __init__(self, urdf_path, device="cpu", *, base_link="base_link", tcp_link="dummy_target",
                 joint_names=None):
        super().__init__()
        self.joint_names = tuple(type(self).joint_names if joint_names is None else joint_names)
        if (not self.joint_names or any(not isinstance(name, str) or not name for name in self.joint_names)
                or len(set(self.joint_names)) != len(self.joint_names)):
            raise ValueError("arm joint_names must be nonempty, unique joint names")
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        self.base_link, self.tcp_link = base_link, tcp_link
        root = ET.parse(self.urdf_path).getroot()
        links = {link.get("name") for link in root.findall("link")}
        if base_link not in links or tcp_link not in links:
            raise ValueError("URDF is missing the requested base or TCP link")
        by_child = {}
        for joint in root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                raise ValueError("URDF joint is missing a parent/child link")
            if child.get("link") in by_child:
                raise ValueError("URDF has multiple parent joints for a link")
            by_child[child.get("link")] = joint
        chain, child, visited = [], tcp_link, set()
        while child != base_link:
            if child in visited or child not in by_child:
                raise ValueError("TCP has no acyclic URDF chain to base_link")
            visited.add(child)
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
        chain.reverse()
        self.chain = []
        moving = []
        for number, joint in enumerate(chain):
            origin = joint.find("origin")
            attrs = origin.attrib if origin is not None else {}
            translation = torch.tensor(_triple(attrs.get("xyz", "0 0 0"), "origin xyz"))
            rpy = _triple(attrs.get("rpy", "0 0 0"), "origin rpy")
            rotation = torch.eye(3)
            # URDF fixed-axis roll/pitch/yaw means Rz(yaw) Ry(pitch) Rx(roll).
            # The float32 construction intentionally matches the existing
            # PoseReferenceTeacher FK used by this repository.
            for axis, angle in zip(([0., 0., 1.], [0., 1., 0.], [1., 0., 0.]), rpy[::-1]):
                rotation = rotation @ _axis_rotation(torch.tensor(axis), torch.tensor([angle]))[0]
            name, kind = joint.get("name"), joint.get("type")
            index = self.joint_names.index(name) if name in self.joint_names else None
            if index is None and kind != "fixed":
                raise ValueError(f"unexpected moving joint in TCP chain: {name}")
            if index is not None:
                if kind not in ("revolute", "continuous"):
                    raise ValueError(f"arm joint {name} is not rotational")
                axis_element = joint.find("axis")
                # The URDF default rotational axis is +X.
                axis = torch.tensor(_triple("1 0 0" if axis_element is None
                                            else axis_element.get("xyz", "1 0 0"), "joint axis"))
                if torch.linalg.vector_norm(axis) < 1.0e-9:
                    raise ValueError(f"zero rotation axis for joint {name}")
                self.register_buffer(f"axis_{number}", axis)
                moving.append(name)
            self.register_buffer(f"origin_R_{number}", rotation)
            self.register_buffer(f"origin_p_{number}", translation)
            self.chain.append((number, name, index))
        if sorted(moving) != sorted(self.joint_names):
            raise ValueError(f"TCP chain must contain the configured arm joints exactly once: {self.joint_names}")
        self.chain_joint_names = tuple(name for _, name, _ in self.chain)
        self.to(device=device)

    def position(self, q: torch.Tensor) -> torch.Tensor:
        q = torch.as_tensor(q, device=self.origin_p_0.device)
        if q.ndim != 2 or q.shape[1] != len(self.joint_names) or not q.is_floating_point():
            raise ValueError(f"encoder angles must be a floating point [N,{len(self.joint_names)}] array "
                             f"in configured joint order {self.joint_names}")
        if not torch.isfinite(q).all():
            raise ValueError("encoder angles contain nonfinite values")
        count = len(q)
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(count, 3, 3)
        position = torch.zeros(count, 3, device=q.device, dtype=q.dtype)
        for number, _, index in self.chain:
            translation = getattr(self, f"origin_p_{number}").to(dtype=q.dtype)
            origin_rotation = getattr(self, f"origin_R_{number}").to(dtype=q.dtype)
            position = position + torch.einsum("nij,j->ni", rotation, translation)
            rotation = rotation @ origin_rotation
            if index is not None:
                axis = getattr(self, f"axis_{number}").to(dtype=q.dtype)
                rotation = rotation @ _axis_rotation(axis, q[:, index])
        return position

    forward = position
