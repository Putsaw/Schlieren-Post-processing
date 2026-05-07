from GUI_functions import *
from clustering import *
from functions_videos import load_cine_video
from data_capture import *

from histogram import *
import opticalFlow as of
import videoProcessingFunctions as vpf
import matplotlib.pyplot as plt
from Legacy.std_functions3 import *
import tkinter as tk
from tkinter import filedialog
import json
import os
from geometry import *


# TODO:
    # optimize performance for larger videos (CUDA?)
    # improve GUI for mask drawing and parameter tuning
    # Maybe standardize naming for videos to use the best settings automatically (TBD)
    # Handle edge cases better, e.g. no motion detected, very short videos, etc?
    # Handle multi file processing better, maybe with a progress bar or batch processing mode?
        # nozzle selection process
        # Combining data
    # Add cone angle display to overlay video
    # Add more metrics to output CSV (spray area, tip velocity, nozzle opening/closing time, etc)


# Hide the main tkinter window
root = tk.Tk()
root.withdraw()



################################
# Load Video Files
###############################
all_files = filedialog.askopenfilenames(title="Select one or more files")
for file in all_files:
    print("Processing:", file)
    video = load_cine_video(file)  # Ensure load_cine_video is defined or imported

    nframes, height, width = video.shape[:3]
    dtype = video[0].dtype
    
    for i in range(nframes):
        frame = video[i]
        if np.issubdtype(dtype, np.integer):
            # For integer types (e.g., uint16): scale down from 16-bit to 8-bit.
            frame_uint8 = (frame / 16).astype(np.uint8)
        elif np.issubdtype(dtype, np.floating):
            # For float types (e.g., float32): assume values in [0,1] and scale up to 8-bit.
            frame_uint8 = np.clip((frame * 255), 0, 255).astype(np.uint8)
            # Boolean case: map False→0, True→255
        elif np.issubdtype(dtype, np.bool_):
            # logger.debug("Frame %d: boolean dtype; converting to uint8", i)
            # Convert bool→uint8 and apply gain, then clip
            frame_uint8 = np.clip(frame.astype(np.uint8) * 255, 0, 255).astype(np.uint8)
        else:
            # Fallback for any other type
            frame_uint8 = (frame / 16).astype(np.uint8)

        video[i] = frame_uint8
    video = video.astype(np.uint8)

    ##############################
    # Video Rotation and Stripping
    ##############################
    rotation_angle = 60  # degrees clockwise, adjust as needed
    rotated_video = vpf.createRotatedVideo(video, rotation_angle) # Rotate 60 degrees clockwise
    firstFrameNumber = vpf.findFirstFrame(rotated_video, threshold=10) # Find first frame with intensity above threshold

    spray_origin = set_spray_origin(file, rotated_video, firstFrameNumber, nframes, height) # Get spray origin from user input or saved data, expects (x, y) format

    if True: # Set to False to disable stripping and use full rotated video
        video_strip = vpf.createVideoStrip(rotated_video, spray_origin, strip_half_height=200)
        nframes, height, width = video_strip.shape[:3]
        # After stripping, reset spray origin to center vertically while keeping x unchanged
        spray_origin = (spray_origin[0], height // 2)
    else:
        video_strip = rotated_video
        nframes, height, width = video_strip.shape[:3]

    first_frame = video_strip[firstFrameNumber]


    # # TESTING
    # TAGS_segmentation = np.zeros_like(video_strip, dtype=np.uint8)
    # TAGS_segmentation_diff = np.zeros_like(video_strip, dtype=np.float32)

    # # Dynamic TAGS background update:
    # # 1) Start from the last frame before injection.
    # # 2) Segment current frame against current background.
    # # 3) Update background pixels only where current frame is classified as background.
    # bg_init_idx = max(0, firstFrameNumber-1)
    # tags_background = video_strip[bg_init_idx].copy()

    # background_mask_test = vpf.createBackgroundMask(first_frame, threshold=20)

    # for i in range(nframes):
    #     current_frame = video_strip[i]
    #     current_frame[background_mask_test == 0] = 0  # Apply background mask to current frame before segmentation

    #     tags_mask, tags_diff = vpf.tags_segmentation(current_frame, tags_background)

    #     TAGS_segmentation[i] = tags_mask
    #     TAGS_segmentation_diff[i] = tags_diff

    #     # Background class in binary mask is 0, foreground/spray is 255.
    #     background_pixels = tags_mask == 0
    #     tags_background[background_pixels] = current_frame[background_pixels]

    #     cv2.imshow("TAGS Segmentation", TAGS_segmentation[i]) # Display rotated video strip for verification
    #     tags_diff_vis = cv2.normalize(tags_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     cv2.imshow("TAGS Segmentation Diff", tags_diff_vis) # Display rotated video strip for verification
    #     cv2.imshow("Current Frame", current_frame) # Display current frame for verification

    #     key = cv2.waitKey(30) & 0xFF
    #     if key == ord('q'):
    #         break
    #     if key == ord('p'):
    #         cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    video_strip2 = video_strip.copy()  # avoid modifying original rotated video for other processing

    video_strip = vpf.applyCLAHE(video_strip)

    background_mask = vpf.createBackgroundMask(first_frame, threshold=20) # Threshold to remove chamber walls
    cv2.imshow("Background Mask", background_mask) # Display background mask for verification, press any key to continue
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    ##############################
    # Optical Flow Visualization
    ##############################
    use_intensity_only = False  # If True, set w_magnitude=0 and use otsu as thresholding, CONSIDER REMOVING AS CUMULATIVE MASK DOES THE SAME THING BUT BETTER PROBABLY

    use_cumulative_as_mask = True  # if True, use cumulative_mask to restrict areas for intensity score, and set w_magnitude=0,
                                    # effectively using cumulative motion detection as a mask for intensity-based detection
    
    if use_intensity_only:
        print("Using intensity-only mode (no optical flow contribution).")
        mag_array = np.ones_like(video_strip, dtype=np.float32)
    else:
        print("Using combined intensity and optical flow mode.")
        mag_array = of.runOpticalFlowCalculationWeighted(firstFrameNumber, video_strip, method='Farneback')

    
    # mag values above 0.4 are considered motion
    # IDEAS:
    #       For cumulative mask, do a morphological erosion every few frames to restrict to areas with consistent motion
    #
    #       Option to add an automated weight setting system which tries to optimize weights based on some criteria?
    #           Will be a lot of work to implement though.
    #
    #       Add a minimum size threshold to remove small noisy detections, maybe based on the expected size of the spray at different frames
    #
    #       Add a minumum size threshold for the blobs in keep_largest_blob, to avoid keeping a small noisy region in front of the spray
    #           Threshold needs to be set low enough that the cone mask can still detect the spray.
    
    # PROBLEMS:
    #           Maybe values over 0.4 in mag should not be set to 1.0 but a lower value, to reduce the dominance of mag in the combination?
    #
    #           Optical flow causes a small area around the chamber walls to be detected as motion. May need to find a way to remove that.
    #
    #           Reconsider high motion detection method:
    #
    #           Cone mask is broken for HPH2, needs adjustment.
    #
    #           Intensity mask sets corners to 1 due to low intensity, but those are not part of the spray. Maybe add a mask to remove corners.
    #              This limits the extraction of the spray.


    # maybe if magnitude is close to zero then intensity should have more weight? not sure how to implement that nicely though.

    # --- Combine per-pixel intensity, optical-flow magnitude, and freehand mask ---
    # Parameters: weights (normalized internally) and binary threshold on combined score (0.0 - 1.0)

    w_intensity = 0.4   # weight for per-pixel light intensity
    w_magnitude = 0.8  # weight for optical flow magnitude
    w_freehand = 0.1    # weight for freehand mask
    w_cone = 0.6    # weight for cone mask
    intensity_gamma = 3.0  # gamma correction for intensity score to amplify differences in dark areas, higher = more contrast

    
    if use_intensity_only:
        w_magnitude = 0
    if use_cumulative_as_mask:
        w_magnitude = 0
        w_intensity = 2.0

    # Precompute full cone mask once; trim per frame based on penetration
    origin_x, origin_y = spray_origin
    cone_angle_deg = 20  # degrees (renamed to avoid conflict with cone_angle array)
    falloff_angle = 30  # degrees over which it decreases to 0
    min_cone_length = 100  # minimum cone length in pixels
    max_cone_length = max(0, width - origin_x - 1)
    yy_full, xx_full = np.ogrid[:height, :width]
    dx_full = xx_full - origin_x
    dy_full = yy_full - origin_y
    angle_full = np.degrees(np.arctan2(dy_full, dx_full))
    abs_angle_full = np.abs(angle_full)

    full_cone_mask = np.zeros((height, width), dtype=np.float32)
    in_forward = dx_full > 0
    in_main = in_forward & (abs_angle_full <= cone_angle_deg)
    in_falloff = in_forward & (abs_angle_full > cone_angle_deg) & (abs_angle_full <= cone_angle_deg + falloff_angle)
    full_cone_mask[in_main] = 1.0
    full_cone_mask[in_falloff] = 1.0 - (abs_angle_full[in_falloff] - cone_angle_deg) / falloff_angle

    # Prepare combined masks array (final binary masks) and a diagnostic combined score array
    combined_masks = np.zeros_like(video_strip, dtype=np.uint8)
    final_cluster_masks = np.zeros_like(video_strip, dtype=np.uint8)
    intensity_scores = np.zeros_like(video_strip, dtype=np.float32)
    mag_scores = np.zeros_like(video_strip, dtype=np.float32)
    cumulative_masks = np.zeros_like(video_strip, dtype=np.uint8)

    cone_masks = np.zeros_like(video_strip, dtype=np.uint8)

    boundaries = []
    penetration = np.zeros(nframes, dtype=np.float32)
    cone_angle = np.zeros(nframes, dtype=np.float32)
    cone_angle_reg = np.zeros(nframes, dtype=np.float32)
    close_point_distance = np.zeros(nframes, dtype=np.float32)
    # store per-side fitted-line angles (degrees, image coords)
    side_angle_up_arr = np.full(nframes, np.nan, dtype=np.float32)
    side_angle_low_arr = np.full(nframes, np.nan, dtype=np.float32)
    angle_d = rotation_angle  # rotation angle in degrees
    spray_area = np.zeros(nframes, dtype=np.float32)
    spray_volume = np.zeros(nframes, dtype=np.float32)

    # Load freehand mask created earlier by the user (expects single-channel binary image "mask.png")
    freehand_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if freehand_mask is None:
        print("Warning: 'mask.png' not found — proceeding without freehand mask")
        freehand_mask_f = np.zeros((height, width), dtype=np.float32)
    else:
        # Resize to match frames if necessary, keep nearest neighbour to preserve binary nature
        if freehand_mask.shape != (height, width):
            freehand_mask = cv2.resize(freehand_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        # Normalize to 0.0-1.0
        freehand_mask_f = (freehand_mask > 0).astype(np.float32)

    # Normalize weights
    total_w = w_intensity + w_magnitude + w_freehand + w_cone
    norm_intensity = w_intensity / total_w
    norm_magnitude = w_magnitude / total_w
    norm_freehand = w_freehand / total_w
    norm_cone = w_cone / total_w

    eps = 1e-6 # small value to avoid division by zero
    cumulative_mask = np.zeros((height, width), dtype=np.uint8)

    write_masks_started = False
    # Precompute circular ROI around spray origin (radius = 100 px)
    # Consider changing circle to cone shape later
    # create second cone mask with a smaller angle to use as a region of interest for detecting motion near the spray origin, which is used to determine when to start writing masks. This helps avoid noise triggering mask writing before the spray actually starts.
    roi_radius = 100
    yy, xx = np.ogrid[:height, :width]
    circle_mask = (xx - origin_x) ** 2 + (yy - origin_y) ** 2 <= roi_radius ** 2

    nozzle_opening_time = 0  # initialize to 0, will be set to first frame with motion near origin

    for idx in range(nframes):
        # --- Intensity: per-frame robust normalization invariant to lighting ---
        frame = video_strip[idx]
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()
        intensity = frame_gray.astype(np.float32)
        # Use percentile-based clipping (1st/99th) per frame so global brightness/contrast changes are normalized out
        if use_cumulative_as_mask:
            masked_pixels = intensity[cumulative_mask > 0]
            if len(masked_pixels) > 0:
                p_low, p_high = np.percentile(masked_pixels, (1.0, 99.0))
            else:
                p_low, p_high = np.percentile(intensity, (1.0, 99.0))  # fallback if no masked pixels
        else:
            p_low, p_high = np.percentile(intensity, (1.0, 99.0))
        if p_high - p_low > 1e-6:
            intensity_n = (intensity - p_low) / (p_high - p_low)
            intensity_n = np.clip(intensity_n, 0.0, 1.0)
            # invert so darker pixels -> higher score
            intensity_n = 1.0 - intensity_n
            # apply gamma to amplify differences in dark areas
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)
        else:
            # fallback to absolute inversion if percentiles degenerate
            intensity_n = 1.0 - (np.clip(intensity, 0.0, 255.0) / 255.0)
            intensity_n = np.clip(intensity_n ** intensity_gamma, 0.0, 1.0)

        # Restrict intensity score to areas within cumulative_mask
        if use_cumulative_as_mask:
            intensity_n[cumulative_mask == 0] = 0

        intensity_scores[idx] = intensity_n

        # --- Optical flow magnitude: cap at mag_clip then normalize to 0..1 (values >= mag_clip -> 1) ---
        mag = mag_array[idx].astype(np.float32)
        mag_clip = 0.4  # absolute motion cutoff: anything higher considered motion and mapped to 1.0
        mag_clipped = np.clip(mag, 0.0, mag_clip)
        mag_n = mag_clipped / (mag_clip + eps)

        mag_scores[idx] = mag_n

        # Accumulate areas with mag_n == 1.0
        new_areas = (mag_n > 0.99).astype(np.uint8) * 255
        cumulative_mask = np.maximum(cumulative_mask, new_areas)
        # cumulative_mask = cv2.erode(cumulative_mask, np.ones((5,5), np.uint8), iterations=1)  # erode to keep only consistent areas

        cumulative_masks[idx] = cumulative_mask.copy()

        # Check for high magnitude values near the spray origin to start writing masks
        if idx >= firstFrameNumber and not write_masks_started:
            # Needs additional code to avoid noise.
            motion_near_origin = vpf.calculate_opening_point(circle_mask, mag)
            if motion_near_origin:
                write_masks_started = True
                nozzle_opening_time = idx  # set to first frame with motion near origin
                w_cone = 1.0  # once motion is detected, set cone weight to normal
        else:
            motion_near_origin = False

        # --- Trim precomputed cone mask based on previous frame penetration ---
        if idx > 0:
            cone_length = max(penetration[idx - 1] + 50, min_cone_length)
        else:
            cone_length = min_cone_length + 50
        cone_length = min(cone_length, max_cone_length)

        cone_mask_f = full_cone_mask.copy()
        if cone_length < max_cone_length:
            cutoff_x = int(origin_x + cone_length)
            if cutoff_x + 1 < width:
                cone_mask_f[:, cutoff_x + 1:] = 0.0

        cone_masks[idx] = (cone_mask_f * 255).astype(np.uint8) # for diagnostics, to be removed later

        freehand = freehand_mask_f  # already 0.0 or 1.0

        # --- Cone mask normalized ---
        cone = cone_mask_f  # already 0.0 or 1.0

        # Replace empty freehand (no drawing) with ones so it doesn't zero-out the product
        if np.count_nonzero(freehand) == 0:
            freehand = np.ones_like(freehand, dtype=np.float32)

        # --- Combine: product (agreement) ---
        comp_int = (intensity_n + eps) ** norm_intensity
        comp_motion = (mag_n + eps) ** norm_magnitude
        comp_free = (freehand + eps) ** norm_freehand
        comp_cone = (cone + eps) ** norm_cone

        # Assume components are already in [0,1]; combine as joint probability
        combined_score = comp_int * comp_motion * comp_free * comp_cone
        # Optional: Normalize to [0,1] 
        combined_score = combined_score / np.max(combined_score) if np.max(combined_score) > 0 else combined_score

        # Optional: map combined_score to 0..255 for diagnostics
        combined_255 = np.clip((combined_score * 255.0), 0, 255).astype(np.uint8)

        # --- Dynamic Thresholding ---
        if use_intensity_only or use_cumulative_as_mask:
            # Use Otsu's thresholding
            combined_uint8 = (combined_score * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(combined_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dynamic_threshold = otsu_thresh / 255.0
            threshold_mask = (combined_score >= dynamic_threshold).astype(np.uint8) * 255
        else:
            # Use 80th percentile of peak score
            peak = combined_score.max()
            threshold_mask = (combined_score >= 0.8 * peak).astype(np.uint8) * 255

        combined_score = cv2.GaussianBlur(combined_score, (5,5), 0) # move before thresholding?

        # Exclude background areas
        threshold_mask[background_mask == 0] = 0

        # OPTIONAL morphological cleanup to remove noise
        # kernel = np.ones((5,5), np.uint8)
        # threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
        # threshold_mask = cv2.dilate(threshold_mask, kernel, iterations=1)
        # threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

        if use_intensity_only or use_cumulative_as_mask:
            # skip clustering for intensity-only mode and cumulative mask mode
            final_mask = fill_holes_in_mask(threshold_mask)
            # Keep only largest blob, connects multiple disjoint regions if present, horizontal threshold determines how far apart blobs can be to be considered connected
            final_mask = keep_largest_blob(final_mask, horizontal_threshold=50, spray_origin=spray_origin) 
            final_cluster_masks[idx] = final_mask
        else:
            # --- Clustering to get final clean outline ---
            # CURRENTLY BUGGED, small cluster distance makes cluster way too small. Does not detect properly. 
            # Cluster distance determines how close points have to be to be considered part of the same cluster, higher = larger clusters
            # Alpha determines concaveness of the hull, higher = more convex, infinity would be full convex, lower = more concave, too low = holes
            final_mask = create_cluster_mask(threshold_mask, cluster_distance=20, alpha=30) 
            final_cluster_masks[idx] = final_mask

        # Convert spray_origin from (x, y) to (row, col) format for analyze_boundary
        # Prepare a clean binary mask for boundary extraction
        final_mask_bin = (final_mask > 0).astype(np.uint8) * 255

        # Extract contours (OpenCV returns contours as Nx1x2 with (x,y) coordinates)
        contours, _ = cv2.findContours(final_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contours to the (row, col) = (y, x) format expected by calculate_boundary
        boundary_list = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)  # (N,2) as (x,y)
            pts_yx = np.column_stack((pts[:, 1], pts[:, 0]))  # (y,x)
            boundary_list.append(pts_yx.astype(float))
    

        # Store only after motion is detected near the spray origin (latched)
        if write_masks_started:
            combined_masks[idx] = threshold_mask
            # --- Analyze boundary ---
            # calculate_boundary(boundary, nozzle_x, nozzle_y, angle_d, ax=None)
            # returns: penetration, average_angle, fitted_angle, boundary_pixels, close_point_distance, angle_line_up, angle_line_low
            frame_pen, frame_ang, frame_ang_reg, frame_boundaries, frame_cpd, side_ang_up, side_ang_low = calculate_boundary( # type: ignore
                boundary_list, spray_origin[0], spray_origin[1], angle_d) 
            side_angle_up_arr[idx] = side_ang_up
            side_angle_low_arr[idx] = side_ang_low

        else:
            combined_masks[idx] = np.zeros_like(threshold_mask)
            frame_boundaries = []
            frame_pen = 0.0
            frame_ang = 0.0
            frame_ang_reg = 0.0
            frame_cpd = 0.0

        boundaries.append(frame_boundaries)
        penetration[idx] = frame_pen
        cone_angle[idx] = frame_ang # average angle of boundary points within a certain distance from the penetration tip, relative to the spray origin, in degrees.
        cone_angle_reg[idx] = frame_ang_reg # same as cone_angle but using a line fitted to the boundary points instead of the raw average
        # cone angle is follows the boundary better, but cone_angle_reg follows the angle? better.
        close_point_distance[idx] = frame_cpd
        tip_velocity = np.gradient(penetration) # derivative of penetration over time
        spray_area[idx] = np.sum(final_mask > 0)  # number of pixels in mask

        spray_volume[idx] =  (np.pi/4) * ((spray_area[idx]**2) * penetration[idx])  # very rough estimate of spray volume based on area and penetration, assumes a cylindrical shape
        # real spray volume is equal or larger.

    intensity_values, intensity_smoothed, intensity_derivative, min_frame, min_value = calculate_video_intensity(video_strip, combined_masks)

    
    # Calculate nozzle closing time
    nozzle_closing_time = vpf.calculate_closing_point(close_point_distance, penetration, intensity_values, spray_area)

    print(f"Final masks computed with w_intensity={w_intensity}, w_magnitude={w_magnitude}, w_freehand={w_freehand}, w_cone={w_cone}, intensity_gamma={intensity_gamma}, use_cumulative_as_mask={use_cumulative_as_mask}, dynamic thresholding (Otsu if cumulative mask or intensity-only, else 95th percentile)")

    # Prepare results output paths
    results_dir = os.path.join(os.getcwd(), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    output_base = os.path.basename(file).replace('.cine', '')
    output_csv = os.path.join(results_dir, f"{output_base}_spray_metrics.csv")
    output_video = os.path.join(results_dir, f"{output_base}_overlay.mp4")

    # Show combined masks (press 'q' to quit, 'p' to pause)
    video_writer = None
    video_fps = 30
    for i in range(nframes):
        frame = video_strip[i]
        combined = combined_masks[i]
        cluster = final_cluster_masks[i]
        cone = cone_masks[i]

        overlay = overlay_cluster_outline(frame, cluster)
        # Ensure overlay is 3-channel BGR so colored drawings show up
        if overlay.ndim == 2 or (overlay.ndim == 3 and overlay.shape[2] == 1):
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        # Draw spray origin and penetration line on overlay
        origin_x, origin_y = spray_origin
        tip_x = int(np.clip(origin_x + penetration[i], 0, width - 1))
        tip_y = int(np.clip(origin_y, 0, height - 1))
        cv2.line(overlay, (int(origin_x), int(origin_y)), (tip_x, tip_y), (0, 255, 255), 5)  # yellow line for penetration
        # Vertical line at penetration tip
        tip_half_len = 40
        y1 = int(np.clip(tip_y - tip_half_len, 0, height - 1))
        y2 = int(np.clip(tip_y + tip_half_len, 0, height - 1))
        cv2.line(overlay, (tip_x, y1), (tip_x, y2), (0, 255, 255), 3)
        cv2.circle(overlay, (int(origin_x), int(origin_y)), 4, (0, 0, 255), -1) 

        # Draw averaged/fitted side-lines for cone angle (above / below)
        up_ang = float(side_angle_up_arr[i]) if not np.isnan(side_angle_up_arr[i]) else None
        low_ang = float(side_angle_low_arr[i]) if not np.isnan(side_angle_low_arr[i]) else None
        # choose a visible length for the diagnostic lines
        vis_len = int(max(50, min(width, height, penetration[i] if penetration[i] > 0 else 100)))/2

        def draw_side_line(angle_deg, color):
            if angle_deg is None or np.isnan(angle_deg):
                return
            rad = np.deg2rad(angle_deg)
            # ensure the line points forward (positive x direction). If cos(rad)<0, flip direction.
            if np.cos(rad) < 0:
                rad += np.pi
            ex = int(np.clip(origin_x + np.cos(rad) * vis_len, 0, width - 1))
            ey = int(np.clip(origin_y + np.sin(rad) * vis_len, 0, height - 1))
            cv2.line(overlay, (int(origin_x), int(origin_y)), (ex, ey), color, 3)

        # above = magenta, below = cyan (BGR)
        draw_side_line(up_ang, (255, 0, 255))
        draw_side_line(low_ang, (255, 255, 0))

        # Resize all images to the same size for stacking (e.g., 640x320)
        def resize(img, size=(640, 320)):
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        def ensure_bgr(img):
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        frame_disp = ensure_bgr(resize(frame))
        combined_disp = ensure_bgr(resize(combined))
        cluster_disp = ensure_bgr(resize(cluster))
        intensity_disp = ensure_bgr(resize((intensity_scores[i] * 255).astype(np.uint8)))
        mag_disp = ensure_bgr(resize((mag_scores[i] * 255).astype(np.uint8)))
        cumulative_disp = ensure_bgr(resize(cumulative_masks[i]))
        cone_disp = ensure_bgr(resize(cone))
        freehand_disp = ensure_bgr(resize((freehand_mask_f * 255).astype(np.uint8)))
        overlay_disp = ensure_bgr(resize(overlay))

        # Add labels to each image
        # Use yellow text for visibility on both black and white backgrounds
        text_color = (0, 255, 255)  # BGR for yellow
        # Draw black border first (thicker)
        cv2.putText(frame_disp, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(combined_disp, "Combined Weighted Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(cluster_disp, "Clustered Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(intensity_disp, "Intensity Score", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(mag_disp, "Optical Flow Magnitude", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(cumulative_disp, "Cumulative Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(cone_disp, "Cone Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(freehand_disp, "Freehand Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(overlay_disp, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)

        # Draw yellow text on top (thinner)
        cv2.putText(frame_disp, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(combined_disp, "Combined Weighted Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cluster_disp, "Clustered Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(intensity_disp, "Intensity Score", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(mag_disp, "Optical Flow Magnitude", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cumulative_disp, "Cumulative Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(cone_disp, "Cone Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(freehand_disp, "Freehand Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(overlay_disp, "Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Now stack and show as before
        row1 = np.hstack([frame_disp, combined_disp, cluster_disp])
        row2 = np.hstack([intensity_disp, mag_disp, cumulative_disp])
        row3 = np.hstack([cone_disp, freehand_disp, overlay_disp])
        grid = np.vstack([row1, row2, row3])

        cv2.imshow('All Results', grid)

        if video_writer is None:
            h, w = overlay.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, (w, h))
        video_writer.write(overlay)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()

    # --- Manual validation on a representative frame ---
    compare_frame_idx = min(200, nframes - 1)
    validation_metrics, _ = draw_and_compare_mask_frames(
        video_strip2,
        final_cluster_masks,
        start_frame=compare_frame_idx,
        save_prefix=output_base,
    )
    print(f"Validation metrics (final selected frame {validation_metrics['frame_index']}):")
    for k, v in validation_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


    # --- Plot all CSV metrics in a grid ---
    frames = np.arange(nframes)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

    axes[0, 0].plot(frames, penetration)
    axes[0, 0].set_title("Penetration")
    axes[0, 0].set_ylabel("Pixels")

    axes[0, 1].plot(frames, cone_angle)
    axes[0, 1].set_title("Cone Angle")
    axes[0, 1].set_ylabel("Degrees")

    axes[0, 2].plot(frames, cone_angle_reg)
    axes[0, 2].set_title("Regularized Cone Angle")
    axes[0, 2].set_ylabel("Degrees")

    axes[1, 0].plot(frames, close_point_distance)
    axes[1, 0].set_title("Close Point Distance")
    axes[1, 0].set_ylabel("Pixels")
    axes[1, 0].set_xlabel("Frame Number")

    axes[1, 1].plot(frames, spray_area)
    axes[1, 1].set_title("Spray Area")
    axes[1, 1].set_ylabel("Pixels$^2$")
    axes[1, 1].set_xlabel("Frame Number")

    axes[1, 2].plot(frames, intensity_values, label="Mean")
    axes[1, 2].set_title("Mean Intensity")
    axes[1, 2].set_ylabel("Gray Value")
    axes[1, 2].set_xlabel("Frame Number")
    axes[1, 2].legend()

    axes[1, 3].plot(frames, intensity_derivative)
    axes[1, 3].axvline(min_frame, color='r', linestyle='--', linewidth=1)
    axes[1, 3].set_title("Intensity Derivative")
    axes[1, 3].set_ylabel("Gray Value / Frame")
    axes[1, 3].set_xlabel("Frame Number")

    axes[0, 3].plot(frames, spray_volume)
    axes[0, 3].set_title("Spray Volume Estimate")
    axes[0, 3].set_ylabel("Cubic Pixels")
    axes[0, 3].set_xlabel("Frame Number")

    fig.suptitle("Spray Metrics Over Time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.show()

    # Generate output CSV in a local Results folder
    with open(output_csv, 'w') as f:
        f.write("Frame,Penetration (pixels), Cone Angle (degrees), Regularized Cone Angle (degrees), Close Point Distance (pixels), Spray Area (pixels^2), Mean Intensity, Intensity Derivative, Spray Volume (cubic pixels), Nozzle Opening Time (frames), Nozzle Closing Time (frames)\n")
        for i in range(nframes):
            f.write(f"{i},{penetration[i]},{cone_angle[i]},{cone_angle_reg[i]},{close_point_distance[i]},{spray_area[i]},{intensity_values[i]},{intensity_derivative[i]},{spray_volume[i]},{nozzle_opening_time},{nozzle_closing_time}\n")

print("Processing complete.")