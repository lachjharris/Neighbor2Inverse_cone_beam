import astra
import numpy as np

EPSILON = np.finfo(np.float32).eps


def _astra_proj_conv(projections):
    """
    Convert:
        [angle, detector_row, detector_col]
    to ASTRA:
        [detector_row, angle, detector_col]
    """
    return np.ascontiguousarray(
        np.swapaxes(projections, 0, 1)
    )


def _create_circular_mask(
    shape,
    center=None,
    radius=None,
):
    """
    NumPy equivalent of the toolkit's
    jax_create_circular_mask().
    """
    height, width = shape

    center = center or (
        height // 2,
        width // 2,
    )

    radius = radius or min(
        center[0],
        center[1],
        width - center[0],
        height - center[1],
    )

    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    dist_from_center = np.sqrt(
        (yy - center[0]) ** 2
        + (xx - center[1]) ** 2
    )

    return np.array(
        dist_from_center <= radius,
        dtype=np.bool_,
    )

def astra_cone_from_array(
    projections: np.ndarray,
    total_rotation_deg: float,
    source_origin_m: float,
    origin_det_m: float,
    true_pixel_size_m: float,
    rotation_offset_deg: int | None = None,
    log_correction: bool = True,
    pad_pixels: int | None = None,
    pad_mode: str = "edge",
    short_scan: bool = False,
    horizontal_detector_offset_m: float = 0,
    vertical_detector_offset_m: float = 0,
    horizontal_source_offset_m: float = 0,
    vertical_source_offset_m: float = 0,
    horizontal_sample_offset_m: float = 0,
    vertical_sample_offset_m: float = 0,
    fbp_filter: str | None = None,
) -> np.ndarray:
    """
    Perform 3D CUDA cone-beam FDK reconstruction from an
    in-memory projection array.

    Input projection shape:
        [angle, detector_row, detector_col]

    Returns:
        reconstruction with shape
        [z, y, x]
    """

    if projections.ndim != 3:
        raise ValueError(
            "projections must have shape "
            "[angle, detector_row, detector_col]"
        )

    # Important: make a copy so log correction etc. do not
    # modify the original g1/g2 array.
    projections = np.asarray(
        projections,
        dtype=np.float32,
    ).copy()

    fbp_filter = fbp_filter or "ram-lak"

    # Circular reconstruction mask.
    circular_mask = np.invert(
        _create_circular_mask(
            shape=(
                projections.shape[2],
                projections.shape[2],
            )
        )
    )

    # [angle, row, col]
    # ->
    # [row, angle, col]
    projections = _astra_proj_conv(projections)

    # Optional horizontal detector padding.
    if pad_pixels:
        projections = np.pad(
            projections,
            (
                (0, 0),
                (0, 0),
                (pad_pixels, pad_pixels),
            ),
            mode=pad_mode,
        )

        pad_slice = slice(
            pad_pixels,
            -pad_pixels,
        )

    else:
        pad_slice = slice(None)

    (
        proj_height_pixels,
        num_projections,
        proj_width_pixels,
    ) = projections.shape

    # Match existing reconstructor exactly.
    angles_deg = np.linspace(
        0,
        total_rotation_deg,
        num_projections,
    )

    if rotation_offset_deg:
        np.add(
            angles_deg,
            rotation_offset_deg,
            out=angles_deg,
        )

    angles_rad = np.deg2rad(angles_deg)

    # Protect log correction.
    projections = np.where(
        np.logical_or(
            projections <= 0,
            np.isnan(projections),
        ),
        EPSILON,
        projections,
    )

    if log_correction:
        np.log(
            projections,
            where=(projections > EPSILON),
            out=projections,
        )

        np.negative(
            projections,
            out=projections,
        )

    # Cone-beam geometry.
    magnification = (
        source_origin_m + origin_det_m
    ) / source_origin_m

    voxel_size_m = (
        true_pixel_size_m / magnification
    )

    basis_magnitude = (
        true_pixel_size_m / voxel_size_m
    )

    vectors = np.zeros(
        (num_projections, 12),
        dtype=np.float64,
    )

    for i, theta in enumerate(angles_rad):

        det_hori_basis_vec = (
            np.array([
                np.cos(theta),
                np.sin(theta),
                0,
            ])
            * basis_magnitude
        )

        det_vert_basis_vec = (
            np.array([0, 0, 1])
            * basis_magnitude
        )

        source_position = (
            np.array([
                np.sin(theta),
                -np.cos(theta),
                0,
            ])
            * (
                source_origin_m
                / voxel_size_m
            )
        )

        source_position += (
            horizontal_source_offset_m
            / voxel_size_m
        ) * (
            det_hori_basis_vec
            / basis_magnitude
        )

        source_position += (
            vertical_source_offset_m
            / voxel_size_m
        ) * (
            det_vert_basis_vec
            / basis_magnitude
        )

        detector_position = (
            np.array([
                -np.sin(theta),
                np.cos(theta),
                0,
            ])
            * (
                origin_det_m
                / voxel_size_m
            )
        )

        detector_position += (
            horizontal_detector_offset_m
            / voxel_size_m
        ) * (
            det_hori_basis_vec
            / basis_magnitude
        )

        detector_position += (
            vertical_detector_offset_m
            / voxel_size_m
        ) * (
            det_vert_basis_vec
            / basis_magnitude
        )

        source_position -= (
            horizontal_sample_offset_m
            / voxel_size_m
        ) * (
            det_hori_basis_vec
            / basis_magnitude
        )

        source_position -= (
            vertical_sample_offset_m
            / voxel_size_m
        ) * (
            det_vert_basis_vec
            / basis_magnitude
        )

        detector_position -= (
            horizontal_sample_offset_m
            / voxel_size_m
        ) * (
            det_hori_basis_vec
            / basis_magnitude
        )

        detector_position -= (
            vertical_sample_offset_m
            / voxel_size_m
        ) * (
            det_vert_basis_vec
            / basis_magnitude
        )

        vectors[i] = [
            *source_position,
            *detector_position,
            *det_hori_basis_vec,
            *det_vert_basis_vec,
        ]

    proj_geom = astra.create_proj_geom(
        "cone_vec",
        proj_height_pixels,
        proj_width_pixels,
        vectors,
    )

    vol_geom = astra.create_vol_geom(
        proj_width_pixels,
        proj_width_pixels,
        proj_height_pixels,
    )

    projections = np.ascontiguousarray(
        projections,
        dtype=np.float32,
    )

    proj_data_id = astra.data3d.link(
        "-sino",
        proj_geom,
        projections,
    )

    reconstructions = np.zeros(
        (
            proj_height_pixels,
            proj_width_pixels,
            proj_width_pixels,
        ),
        dtype=np.float32,
    )

    rec_id = astra.data3d.link(
        "-vol",
        vol_geom,
        reconstructions,
    )

    projector_id = astra.create_projector(
        "cuda3d",
        proj_geom,
        vol_geom,
    )

    cfg = astra.astra_dict("FDK_CUDA")

    cfg["ProjectorId"] = projector_id
    cfg["ProjectionDataId"] = proj_data_id
    cfg["ReconstructionDataId"] = rec_id

    cfg["option"] = {
        "ShortScan": short_scan,
        "FilterType": fbp_filter,
    }

    alg_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(alg_id)

    finally:
        astra.algorithm.delete(alg_id)
        astra.projector.delete(projector_id)
        astra.data3d.delete(proj_data_id)
        astra.data3d.delete(rec_id)

    # Same physical normalization as existing function.
    np.divide(
        reconstructions,
        voxel_size_m,
        out=reconstructions,
    )

    # Remove reconstruction padding.
    reconstructions = reconstructions[
        :,
        pad_slice,
        pad_slice,
    ]

    # Apply circular mask.
    reconstructions[:, circular_mask] = 0

    return reconstructions