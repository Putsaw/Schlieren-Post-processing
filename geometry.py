# -----------------------------------------------------------
# This file contains the geometry / measurement logic that runs after a 
# spray boundary is detected. 

# Big picture:
# 1. Take boundary points of the detected spray blob
# 2. Measure how far each boundary point is from the nozzle
# 3. Find the point of maximum penetration (farthest from nozzle)
# 4. Select points that are between 10% and 60% of the penetration distance
    # because that region is more stable for cone angle measurement
# 5) Split those points into the two sides of the spray (upper/lower side)
# 6. Estimate the cone angle in two ways:
#      - Average angle method: average the angles of the selected points on each side.
#      - Regression method: fit a line to the selected points on each side and compute the angle between those lines.
# 7. Optionally draw the helper lines on matplotlib axis.

# Coordinate convertion used here:
#    - Image x increases to the right
#   - Image y increases downwards
#    - Boundary points are in (row, col) = (y, x) format
# -----------------------------------------------------------

from __future__ import annotations

# NumPy is used for almost all numerical operations in this file
import numpy as np

# matplotlib is only used if an axis is passed in for plotting diagnostics
import matplotlib.pyplot as plt


def calculate_boundary(boundary, nozzle_x, nozzle_y, angle_d, ax=None):
    """
    Parameters
    ----------
    boundary : list[np.ndarray] | np.ndarray | None
        Boundary points of the spray in (row, col) = (y, x) format.
        In practice, this often comes from skimage.measure.find_contours(...),
        which returns one or more contour arrays.
    nozzle_x, nozzle_y : float
        Nozzle position in image coordinates.
    angle_d : float
        Spray centerline angle in degrees.
    ax : matplotlib.axes.Axes | None
        Optional matplotlib axis for drawing helper lines and diagnostics.

    Returns
    -------
    penetration : float
        Maximum distance from the nozzle to any boundary point.
    average_angle : float
        Cone angle estimate from average point angles on both sides of the spray.
    fitted_angle : float
        Cone angle estimate based on fitted lines on each side of the spray.
    boundary_pixels : np.ndarray
        All boundary points stacked into one array of shape (N, 2).
    close_point_distance : float
        Minimum distance from the nozzle to the boundary.
    
    Notes   
    -----
    Robust calculate_boundary for image coordinates (x right, y down).

    boundary points are expected in (row, col) = (y, x) format.
    """

    # To simplify the rest of the code, we convert everything to a list of arrays.
    if boundary is None:
        boundary = []

    if isinstance(boundary, np.ndarray):
        boundary = [] if boundary.size == 0 else [boundary]

    if len(boundary) == 0:
        boundary_pixels = np.array([[nozzle_y, nozzle_x]], dtype=float)
        return 0.0, 0.0, 0.0, boundary_pixels, 0.0

     # -------------------------------------------------------------------------
    # 1) Combine all contour pieces into one (N, 2) array
    # -------------------------------------------------------------------------
    # np.vstack stacks arrays vertically.
    # Example: [(100,2), (80,2)] -> (180,2)
    boundary_pixels = np.vstack(boundary).astype(float)

    # Boundary points are in (row, col) order, so:
    # - column 0 is y (vertical position)
    # - column 1 is x (horizontal position)
    ys = boundary_pixels[:, 0]
    xs = boundary_pixels[:, 1]

    # -------------------------------------------------------------------------
    # 2) Distance from nozzle to every boundary point
    # -------------------------------------------------------------------------
    # Shift all points so the nozzle acts like the local origin.
    dx = xs - nozzle_x
    dy = ys - nozzle_y
    
    # Euclidean distance for every point:
    # distance = sqrt(dx^2 + dy^2)
    distances = np.sqrt(dx**2 + dy**2)

    # np.argmax returns the index of the largest value.
    peak_index = int(np.argmax(distances))
    
    # Penetration = farthest boundary point from the nozzle.
    penetration = float(distances[peak_index])
    
    # Closest point distance can be useful as another diagnostic metric.
    close_point_distance = float(np.min(distances))

    # -------------------------------------------------------------------------
    # 3) MATLAB-like helper intersections for plotting only
    # -------------------------------------------------------------------------
    # This section is mostly a diagnostic / visualization aid.
    # It reproduces the style of the original MATLAB helper lines.
    #
    # slope1 is the slope of the spray centerline.
    slope1 = np.tan(np.deg2rad(angle_d))

    denom = 1 + slope1**2
    if np.isclose(denom, 0):
        xinter = nozzle_x
        yinter = nozzle_y
    else:
        xinter = (
            (-slope1 * nozzle_y + slope1 * ys[peak_index] + xs[peak_index] - nozzle_x)
            / denom
        ) + nozzle_x
        yinter = slope1 * (xinter - nozzle_x) + nozzle_y

    show_ang = 20
    slope2 = np.tan(np.deg2rad(angle_d + show_ang))
    slope3 = np.tan(np.deg2rad(angle_d - show_ang))

    denom1 = 1 + slope1 * slope2
    denom2 = 1 + slope1 * slope3

    if np.isclose(denom1, 0):
        xinter1 = nozzle_x
        yinter1 = nozzle_y
    else:
        xinter1 = (
            slope1 * (yinter - nozzle_y) + slope1 * slope2 * nozzle_x + xinter
        ) / denom1
        yinter1 = slope2 * (xinter1 - nozzle_x) + nozzle_y

    if np.isclose(denom2, 0):
        xinter2 = nozzle_x
        yinter2 = nozzle_y
    else:
        xinter2 = (
            slope1 * (yinter - nozzle_y) + slope1 * slope3 * nozzle_x + xinter
        ) / denom2
        yinter2 = slope3 * (xinter2 - nozzle_x) + nozzle_y

     # -------------------------------------------------------------------------
    # 4) Keep only a useful distance range for cone-angle calculation
    # -------------------------------------------------------------------------
    # Near the nozzle, the boundary is often noisy or constrained by hardware.
    # Near the spray tip, the plume can become bulbous or unstable.
    #
    # A common practical choice is to estimate the cone angle from the middle part
    # of the plume. Here: 10% to 60% of total penetration.
    lower_bound = 0.1 * penetration
    upper_bound = 0.6 * penetration
    angle_ind = np.where((distances > lower_bound) & (distances < upper_bound))[0]

    if angle_ind.size == 0:
        # return consistent shape/types even when no angle-range points found
        angle_line_up = np.nan
        angle_line_low = np.nan
        average_angle = np.nan
        fitted_angle = np.nan
        return penetration, average_angle, fitted_angle, boundary_pixels, close_point_distance, angle_line_up, angle_line_low

    angx = xs[angle_ind]
    angy = ys[angle_ind]

    # -------------------------------------------------------------------------
    # 5) Simple linear-side cone-angle method (replaced previous complex logic)
    # -------------------------------------------------------------------------
    # New approach:
    # - Take points between 10% and 60% penetration (angx, angy)
    # - Split them by vertical position relative to the nozzle (above / below)
    # - Fit a straight line (y = m*x + b) to each side if possible
    # - Compute each side's angle (deg) from the horizontal and the relative
    #   deviation from the provided centerline angle_d
    # - average_angle: sum of the two side deviations when both present, otherwise
    #   the single available side; fitted_angle: geometric angle between the two
    #   fitted lines (if both available)
    mask_above = angy < nozzle_y
    mask_below = angy >= nozzle_y

    def fit_line_angle(hx, hy):
        if hx.size < 2:
            return np.nan  # angle_line_deg
        # fit y = m*x + b
        m, b = np.polyfit(hx, hy, 1)
        angle_line = float(np.degrees(np.arctan(m)))  # line angle in degrees (image coords)
        return angle_line

    angle_line_up = fit_line_angle(angx[mask_above], angy[mask_above])
    angle_line_low = fit_line_angle(angx[mask_below], angy[mask_below])

    # Compute fitted_angle as geometric angle between two fitted lines (if both exist)
    if not np.isnan(angle_line_up) and not np.isnan(angle_line_low):
        m_up = np.tan(np.deg2rad(angle_line_up))
        m_low = np.tan(np.deg2rad(angle_line_low))
        fitted_angle = float(np.degrees(np.arctan2(abs(m_up - m_low), 1.0 + m_up * m_low)))
    else:
        fitted_angle = np.nan

    # --- New: average_angle computed purely from straight lines from nozzle -> mean boundary point (no angle_d) ---
    # For each side compute the vector from nozzle to the centroid of the side points,
    # then compute the included angle between those two centroid vectors.
    def centroid_angle(hx, hy):
        if hx.size == 0:
            return np.nan
        cx = float(np.mean(hx))
        cy = float(np.mean(hy))
        ang = float(np.degrees(np.arctan2(cy - nozzle_y, cx - nozzle_x)))  # image coords
        return ang

    cent_ang_up = centroid_angle(angx[mask_above], angy[mask_above])
    cent_ang_low = centroid_angle(angx[mask_below], angy[mask_below])

    if np.isnan(cent_ang_up) or np.isnan(cent_ang_low):
        average_angle = np.nan
    else:
        diff = abs(cent_ang_up - cent_ang_low) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        average_angle = float(diff)

    # --- Optional plotting of cone angle lines ---
    #return penetration, average_angle, fitted_angle, boundary_pixels, close_point_distance, angle_line_up, angle_line_low
    return penetration, average_angle, fitted_angle, boundary_pixels, close_point_distance, cent_ang_up, cent_ang_low


def calculate_video_intensity(video_strip, combined_masks):
   # --- Analyze intensity values ---
    # Needs more work, maybe diffent method to find significant changes
    nframes = video_strip.shape[0]
    intensity_values = np.zeros(nframes, dtype=np.float32)
    for i in range(nframes):
        masked_pixels = video_strip[i][combined_masks[i] > 0]
        intensity_values[i] = float(masked_pixels.mean()) if masked_pixels.size > 0 else 0.0

    # Keep smoothed intensity aligned to the per-frame metrics for plotting/export.
    window_size = 5
    if intensity_values.size == 0:
        intensity_smoothed = np.zeros(0, dtype=np.float32)
        intensity_derivative = np.zeros(0, dtype=np.float32)
    else:
        smoothing_kernel = np.ones(window_size, dtype=np.float32) / window_size
        intensity_smoothed = np.convolve(intensity_values, smoothing_kernel, mode='same')
        intensity_derivative = np.diff(intensity_smoothed, prepend=intensity_smoothed[0])

    frame_numbers = np.arange(nframes)

    # only consider frames at least 10 after firstFrameNumber
    start_offset = 10
    end_offset = 10
    if len(intensity_derivative) <= start_offset:
        # not enough frames — fall back to full range
        min_idx = int(np.argmin(intensity_derivative))
    else:
        sliced = intensity_derivative[start_offset:]
        rel_min = int(np.argmin(sliced))
        min_idx = rel_min + start_offset

    min_frame = int(frame_numbers[min_idx])
    min_value = float(intensity_derivative[min_idx])
    print(f"Lowest intensity derivative at frame {min_frame} (index {min_idx}) = {min_value:.6f}")

    return intensity_values, intensity_smoothed, intensity_derivative, min_frame, min_value