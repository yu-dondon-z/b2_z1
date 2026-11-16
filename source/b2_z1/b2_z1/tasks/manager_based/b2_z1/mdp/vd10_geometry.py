"""Analytic outer-cylinder clearance for the measured VD10 welding tube."""

import torch


def cylinder_tip_height_above_plane(axis_w, clearance=0.0, length=1.1, radius=0.028):
    """Height of the front end-face centre for the whole cylinder to clear z=0.

    The unit axis points from the rear to the TCP. A tilted end face extends
    below its centre by r*sqrt(1-axis_z**2); an upward axis also lowers the rear.
    This is geometry only. It neither changes nor estimates contact forces.
    """
    axis_z = axis_w[..., 2].clamp(-1.0, 1.0)
    radial_extent = radius * torch.sqrt((1.0 - axis_z.square()).clamp_min(0.0))
    return (length * axis_z).clamp_min(0.0) + radial_extent + clearance


def cylinder_surface_min_z(tip_w, axis_w, length=1.1, radius=0.028):
    """Lowest world z of the finite cylinder's physical outer surface."""
    return tip_w[..., 2] - cylinder_tip_height_above_plane(axis_w, length=length, radius=radius)
