"""Rear RGB-D housing and a sensor-safe viewport field-of-view overlay.

The lightweight housing is only a schematic D435i envelope, not a calibrated
CAD model.  Its local coordinates are ROS optical (+X right, +Y down, +Z
forward), exactly like the camera offset.  Every surface lies behind Z=0 so
the camera cannot see its own housing.  No physics or mass API is authored.

The frustum is an ``omni.ui.scene`` overlay owned by this helper, rather than
USD geometry or Isaac's shared debug-draw buffer.  It is visible in the user
viewport, cannot enter camera RGB/depth images and never clears other lines.
Its 1.5 m far rectangle is a display length, not the sensor depth limit.
"""

from __future__ import annotations

import math

import numpy as np


MODEL_SIZE_M = (0.090, 0.025, 0.024)
MODEL_FRONT_Z_M = -0.004


def _rotation_wxyz(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("camera quaternion must be a finite nonzero wxyz vector")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def frustum_segments_world(intrinsic_matrix, width, height, position_w, quaternion_wxyz,
                           near_m=0.10, far_m=1.50):
    """Return 17 world-space segments from real K and a ROS optical pose.

    Pixel-edge rays preserve off-center principal points and unequal focal
    lengths.  The near/far values are optical-Z distances, like image-plane
    depth, rather than radial range.  The last segment is the optical axis.
    This pure NumPy function can also support independent scene diagnostics.
    """
    k = np.asarray(intrinsic_matrix, dtype=np.float64)
    position = np.asarray(position_w, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("camera intrinsic matrix must be finite and nonsingular")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera position must be a finite XYZ vector")
    if (int(width) != width or int(height) != height or width < 1 or height < 1
            or not math.isfinite(near_m) or not math.isfinite(far_m)
            or not 0 < near_m < far_m):
        raise ValueError("invalid frustum image dimensions or display distances")
    rotation = _rotation_wxyz(quaternion_wxyz)
    # Intrinsics use pixel centers. These are the four outer image edges.
    pixels = np.array([
        [-0.5, -0.5, 1.0], [width - 0.5, -0.5, 1.0],
        [width - 0.5, height - 0.5, 1.0], [-0.5, height - 0.5, 1.0],
    ])
    rays = pixels @ np.linalg.inv(k).T
    if np.any(np.abs(rays[:, 2]) < 1e-12):
        raise ValueError("intrinsics produce a ray parallel to the image plane")
    rays /= rays[:, 2:3]
    near = position + near_m * rays @ rotation.T
    far = position + far_m * rays @ rotation.T
    segments = []
    for index in range(4):
        following = (index + 1) % 4
        segments.extend(((near[index], near[following]), (far[index], far[following]),
                         (near[index], far[index]), (position, near[index])))
    segments.append((position, position + rotation[:, 2] * far_m))
    return np.asarray(segments)


def _spawn_d435i_model_impl(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, UsdGeom, UsdShade
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"camera housing already exists: {prim_path}")
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    rotation = orientation or (1.0, 0.0, 0.0, 0.0)
    root.AddOrientOp().Set(Gf.Quatf(float(rotation[0]), Gf.Vec3f(*rotation[1:])))

    def material(name, color, metallic=0.0):
        mat = UsdShade.Material.Define(stage, f"{prim_path}/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        from pxr import Sdf
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    silver = material("SilverHousing", (0.58, 0.62, 0.65), 0.65)
    black = material("BlackOpticalFace", (0.018, 0.022, 0.028))
    glass = material("BlueLensGlass", (0.035, 0.10, 0.18), 0.35)

    def bind(shape, mat):
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)

    def box(name, center, size, mat):
        shape = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        shape.CreateSizeAttr(2.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
        shape.AddScaleOp().Set(Gf.Vec3f(*(value * 0.5 for value in size)))
        bind(shape, mat)

    box("Housing", (0.0, 0.0, MODEL_FRONT_Z_M - MODEL_SIZE_M[2] * 0.5), MODEL_SIZE_M, silver)
    box("OpticalFace", (0.0, 0.0, -0.003), (0.084, 0.021, 0.002), black)
    # Three separate front windows make the stereo RGB-D module legible.
    # Their foremost surface remains 1 mm BEHIND the actual optical origin.
    for name, x, radius in (("LeftInfrared", -0.029, 0.0055),
                             ("Projector", 0.0, 0.0040),
                             ("RightRgb", 0.029, 0.0055)):
        lens = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
        lens.CreateAxisAttr(UsdGeom.Tokens.z)
        lens.CreateRadiusAttr(radius)
        lens.CreateHeightAttr(0.002)
        lens.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, -0.002))
        bind(lens, glass)
    root.GetPrim().SetCustomDataByKey("camera_visual:optical_convention", "ros")
    root.GetPrim().SetCustomDataByKey("camera_visual:physics", "visual_only_no_mass_or_collision")
    return root.GetPrim()


def _spawn_d435i_model(prim_path, cfg, translation=None, orientation=None, **kwargs):
    # Defer all Omniverse imports so the geometry functions remain usable in
    # CPU tests outside SimulationApp; use IsaacLab's supported regex cloner.
    from isaaclab.sim.utils import clone
    return clone(_spawn_d435i_model_impl)(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


def make_camera_model_cfg(prim_path, position, quaternion_wxyz):
    """AssetBaseCfg for a sibling of the camera prim under ``base_link``.

    Pass the same ROS optical offset as CameraCfg.  This schematic model
    follows the base through USD parenting and creates no rigid body, mass,
    collision geometry or contact sensor.
    """
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SpawnerCfg

    _rotation_wxyz(quaternion_wxyz)
    return AssetBaseCfg(
        prim_path=prim_path, spawn=SpawnerCfg(func=_spawn_d435i_model),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position), rot=tuple(quaternion_wxyz)),
    )


class RearCameraViewportVisual:
    """Own one viewport overlay; call ``update`` after reading camera data.

    In headless mode (no viewport window), update returns False and creates
    no geometry or global drawing state.  A later GUI viewport can attach on
    the next call.  Near/far are visualization distances, independent of the
    camera clipping range.  Call ``close`` during environment shutdown.
    """

    def __init__(self, camera, near_m=0.10, far_m=1.50, env_index=0):
        if not 0 < near_m < far_m:
            raise ValueError("invalid camera frustum display distances")
        self.camera = camera
        self.near_m = near_m
        self.far_m = far_m
        self.env_index = env_index
        self._viewport_window = None
        self._scene_view = None
        self._frame = None
        self._frame_name = f"b2_z1.va16.rear_camera_frustum.{id(self)}"
        self._sc = None
        self._closed = False

    @property
    def available(self):
        return self._scene_view is not None

    def _attach(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            from omni.ui import scene as sc
        except ImportError:
            # Headless applications may omit the optional viewport/UI
            # extensions entirely. Sensor export must keep working there.
            return False

        viewport = get_active_viewport_window()
        if viewport is None:
            return False
        self._viewport_window = viewport
        self._sc = sc
        self._frame = viewport.get_frame(self._frame_name)
        with self._frame:
            self._scene_view = sc.SceneView(opaque_for_mouse_events=False)
        viewport.viewport_api.add_scene_view(self._scene_view)
        return True

    def update(self):
        if self._closed:
            return False
        if self._scene_view is None and not self._attach():
            return False
        import torch
        # Playback runs under inference_mode. A first odd-step UI read must
        # not allocate inference tensors that the normal camera exporter then
        # cannot update outside that mode on its next scheduled capture.
        with torch.inference_mode(False):
            data = self.camera.data
            index = self.env_index
            position = data.pos_w[index].detach().cpu().numpy()
            quaternion = data.quat_w_ros[index].detach().cpu().numpy()
            intrinsic = data.intrinsic_matrices[index].detach().cpu().numpy()
            height, width = data.image_shape
        segments = frustum_segments_world(
            intrinsic, width, height, position, quaternion, self.near_m, self.far_m)
        # This private SceneView contains only this overlay's own lines.
        self._scene_view.scene.clear()
        with self._scene_view.scene:
            for segment in segments[:-1]:
                self._sc.Line(segment[0].tolist(), segment[1].tolist(),
                              color=(0.0, 0.72, 1.0, 0.82), thickness=2)
            axis = segments[-1]
            self._sc.Line(axis[0].tolist(), axis[1].tolist(),
                          color=(0.15, 0.9, 1.0, 0.9), thickness=2)
        return True

    def close(self):
        if self._closed:
            return
        # Close is terminal, including when called by a finalizer. A later
        # simulation/update callback must not reattach to a closing viewport.
        self._closed = True
        scene_view, self._scene_view = self._scene_view, None
        viewport, self._viewport_window = self._viewport_window, None
        frame, self._frame = self._frame, None
        # Camera.__del__ detaches its Replicator annotators. Do not let this
        # UI helper retain the sensor beyond environment/simulator shutdown.
        self.camera = None
        self._sc = None
        if scene_view is not None:
            scene_view.scene.clear()
            if viewport is not None:
                viewport_api = viewport.viewport_api
                if viewport_api is not None:
                    viewport_api.remove_scene_view(scene_view)
            scene_view.destroy()
        if frame is not None:
            # The viewport owns the named Frame; clear only our own children
            # rather than destroying a Frame still registered with it.
            frame.clear()
