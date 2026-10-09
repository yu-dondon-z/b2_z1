"""Local planar path geometry, independent of Isaac Sim imports."""
import torch


def cached_local_geometry(points, half_window=0.5, minimum_span=0.3):
    """Fit the previously seen path near the base; return None if unavailable.

    Cache insertion order is not path order, so do not connect its raw points.
    A small least-squares y(x) fit suppresses pixel/depth segment jitter.
    """
    local = points[points[:, 0].abs() <= half_window, :2]
    if len(local) < 4 or float(local[:, 0].max() - local[:, 0].min()) < minimum_span:
        return None
    centre = local.mean(dim=0)
    delta = local - centre
    slope = (delta[:, 0] * delta[:, 1]).sum() / delta[:, 0].square().sum().clamp_min(1.0e-6)
    norm = torch.sqrt(1.0 + slope.square())
    tangent = torch.stack((1.0 / norm, slope / norm))
    lateral = (centre[1] - slope * centre[0]) / norm
    return tangent, lateral


def local_path_geometry(points):
    """Return forward tangent and signed origin-to-path normal offset.

    Points must be ordered along the visible path. If the camera cannot see
    the base projection, extrapolate only the closest endpoint segment rather
    than treating the distant lookahead's y coordinate as cross-track error.
    """
    starts = points[:-1, :2]
    segments = points[1:, :2] - starts
    lengths = torch.linalg.vector_norm(segments, dim=1)
    valid = lengths > 1.0e-5
    if not bool(valid.any()):
        raise ValueError("Path has no nonzero planar segments")
    starts, segments, lengths = starts[valid], segments[valid], lengths[valid]
    fractions = -(starts * segments).sum(dim=1) / lengths.square()
    nearest = starts + fractions.clamp(0.0, 1.0).unsqueeze(1) * segments
    index = nearest.square().sum(dim=1).argmin()
    tangent = segments[index] / lengths[index]
    tangent = torch.where(tangent[0] < 0.0, -tangent, tangent)
    # Normal projection is unchanged by extrapolation along this segment.
    lateral = -tangent[1] * starts[index, 0] + tangent[0] * starts[index, 1]
    return tangent, lateral
