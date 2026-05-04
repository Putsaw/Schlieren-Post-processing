################################
# Removes background from a frame using the first frame as reference
################################
from matplotlib import image


def removeBackgroundSimple(video, first_frame, threshold=10):
    import cv2
    import numpy as np

    nframes = video.shape[0]
    foreground_video = np.zeros_like(video)

    for i in range(nframes):
        frame = video[i]

        # Compute absolute difference between the frame and the background
        diff = cv2.absdiff(frame, first_frame)

        # Threshold the grayscale difference to create a binary mask
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Use the mask to extract the foreground from the original frame
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        foreground_video[i] = foreground

    return foreground_video


def createBackgroundMask(first_frame, threshold=10):
    import numpy as np

    # Binary mask: 1 where difference is greater than threshold, else 0 (uint8)
    mask = (first_frame > threshold).astype(np.uint8) * 255

    return mask


################################
# Rotates each frame in the video by a specified angle
################################
def createRotatedVideo(video, angle):
    import cv2
    import numpy as np

    nframes, height, width = video.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_video = np.zeros_like(video)
    for i in range(nframes):
        rotated_frame = cv2.warpAffine(video[i], rotation_matrix, (width, height))
        rotated_video[i] = rotated_frame

        # experimental - try different interpolation and border modes
        # same results as above
        # rotated_frame = cv2.warpAffine(
        #     video[i],
        #     rotation_matrix,
        #     (width, height),
        #     flags=cv2.INTER_NEAREST,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=0,
        # )
        # rotated_video[i] = rotated_frame

    return rotated_video

################################
# Creates a video strip by extracting a symmetric band around spray_origin
################################
def createVideoStrip(video, spray_origin, strip_half_height=200):
    import numpy as np

    nframes, height, width = video.shape

    # Ensure integer row index
    if isinstance(spray_origin, (tuple, list, np.ndarray)) and len(spray_origin) >= 2:
        origin_row = spray_origin[1]
    else:
        origin_row = spray_origin
    origin_row = int(round(origin_row)) # type: ignore
    origin_row = max(0, min(height - 1, origin_row))

    # Compute maximum symmetric half-height possible
    max_above = min(strip_half_height, origin_row)
    max_below = min(strip_half_height, (height - 1) - origin_row)
    half_height = min(max_above, max_below)

    start_row = origin_row - half_height
    end_row = origin_row + half_height + 1

    strip_height = end_row - start_row
    video_strip = np.zeros((nframes, strip_height, width), dtype=video.dtype)

    for i in range(nframes):
        video_strip[i] = video[i, start_row:end_row, :]

    return video_strip

###############################
# Takes a video and returns the first frame with mean intensity above a threshold
################################
def findFirstFrame(video, threshold=10):
    import numpy as np
    nframes = video.shape[0]

    for i in range(1, nframes):
        frame = video[i]
        # Compute mean brightness
        mean_intensity = frame.mean()

        if mean_intensity > threshold:
            return i
        
    return 0  # Default to first frame if no suitable frame is found

################################
# Applies a binary threshold to each frame in the video, UNUSED
################################
def removeBackgroundThreshold(video, threshold=30):
    #Consider making frame specific
    import cv2
    import numpy as np
    
    nframes = video.shape[0]
    for i in range(nframes):
        frame = video[i]
        _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        video[i] = frame

    return video

################################
# Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to each frame in the video
################################
def applyCLAHE(video, clipLimit=2.0, tileGridSize=(8,8)):
    import cv2
    import numpy as np

    nframes = video.shape[0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    for i in range(nframes):
        frame = video[i]
        frame = clahe.apply(frame)
        video[i] = frame

    return video

def applyLaplacianFilter(video):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.Laplacian(frame, cv2.CV_64F)
        frame = cv2.convertScaleAbs(frame)
        video[i] = frame

    return video

def applyDoGfilter(video, low_sigma=1.0, high_sigma=2.0):
    from skimage.filters import difference_of_gaussians
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]

        # DoG output is float and can contain negative values; convert to displayable uint8.
        dog = difference_of_gaussians(frame, low_sigma=low_sigma, high_sigma=high_sigma)
        dog = np.abs(dog)

        max_val = dog.max()
        if max_val > 0:
            dog = (dog / max_val) * 255.0
        else:
            dog = np.zeros_like(dog)

        video[i] = dog.astype(np.uint8)

    return video

def adaptiveGaussianThreshold(video, maxValue=255, blockSize=11, C=2):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = cv2.adaptiveThreshold(frame, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize, C)
        video[i] = frame

    return video


def OtsuThreshold(video, background_mask):
    import cv2
    import numpy as np

    out = video.copy()  
    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]

        # Otsu binarization
        _, otsu_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Alternative fixed thresholding
        # _, otsu_frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

        if background_mask is None:
            # No mask provided: return Otsu result
            foreground = otsu_frame
        else:
            bg = background_mask

            # Normalize mask to uint8 0/255 format for XOR
            if bg.dtype != np.uint8:
                bg = bg.astype(np.uint8)
            if bg.max() == 1:
                bg = bg * 255

            # If a single mask (not per-frame) was passed, allow it to be used for all frames
            # If a per-frame list/array was passed, pick the corresponding frame mask
            if isinstance(bg, (list, tuple)):
                bg_frame = bg[i]
            elif bg.ndim == 3 and bg.shape[0] == nframes:
                bg_frame = bg[i]
            else:
                bg_frame = bg

            # Resize mask if necessary to match frame shape
            if bg_frame.shape != otsu_frame.shape:
                bg_frame = cv2.resize(bg_frame, (otsu_frame.shape[1], otsu_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # XOR Otsu result with background mask
            foreground = cv2.bitwise_xor(otsu_frame, bg_frame)

        out[i] = foreground

    return out



def invertVideo(video):
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        frame = 255 - frame
        video[i] = frame

    return video


# chan-vese segmentation is too slow, consider other methods
def chanVeseSegmentation(video):
    from skimage.segmentation import chan_vese
    import numpy as np
    import cv2
    from skimage import img_as_float # type: ignore

    nframes = video.shape[0]


    for i in range(120, 150): # Limiting to frames 120-150 for speed during testing
        frame = video[i]

        # Convert to float
        img_f = img_as_float(frame)

        # Smooth to suppress schlieren texture
        img_smooth = cv2.GaussianBlur(img_f, (0, 0), 5)

        # Apply Chan-Vese segmentation
        cv_result = chan_vese(
                img_smooth,
                mu=0.3,                # contour smoothness
                lambda1=1.0,           # inside-region weight
                lambda2=1.0,           # outside-region weight
                tol=1e-3,
                max_num_iter=200,
                dt=0.5,
                init_level_set="checkerboard"
        )
        cv_result = (cv_result * 255).astype(np.uint8)  # type: ignore

        video[i] = cv_result

    return video

def temporalMedianFilter(video, firstFrameNumber):
    import cv2
    import numpy as np

    frames = np.stack(video[firstFrameNumber:firstFrameNumber+30], axis=0)
    background = np.median(frames, axis=0)

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        diff = np.abs(frame - background)
        mask = diff

        video[i] = mask

    return video


# Adaptive background subtraction with online update
# only really works well for videos with static background
# which would probably work better with simpler methods
def adaptive_background_subtraction(
    video,
    start=0,
    n_bg_frames=10,
    alpha=0.01,
    blur_sigma=3,
    thresh_percentile=80
):
    import numpy as np
    import cv2

    # ---- INITIAL BACKGROUND (temporal median) ----
    bg_frames = video[start:start + n_bg_frames]
    bg_frames = np.array([
        cv2.GaussianBlur(f, (0,0), blur_sigma) for f in bg_frames
    ])
    background = np.median(bg_frames, axis=0).astype(np.float32)

    plume_masks = []
    eps = 1e-3

    for frame in video:
        # ---- Preprocess ----
        frame_blur = cv2.GaussianBlur(frame, (0,0), blur_sigma).astype(np.float32)

        # ---- Normalized intensity difference ----
        diff_i = cv2.absdiff(frame_blur, background)
        diff_i /= (background + eps)

        # ---- Gradient difference ----
        gx_f = cv2.Sobel(frame_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy_f = cv2.Sobel(frame_blur, cv2.CV_32F, 0, 1, ksize=3)

        gx_b = cv2.Sobel(background, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(background, cv2.CV_32F, 0, 1, ksize=3)

        diff_g = np.sqrt((gx_f - gx_b)**2 + (gy_f - gy_b)**2)

        # ---- Combined foreground score ----
        score = diff_i + 0.7 * diff_g

        # ---- Threshold (percentile-based) ----
        thresh = np.percentile(score, thresh_percentile)
        plume_mask = score > thresh

        plume_masks.append(plume_mask.astype(np.uint8) * 255)

        # ---- ADAPTIVE BACKGROUND UPDATE (masked) ----
        background_mask = ~plume_mask
        background[background_mask] = (
            (1 - alpha) * background[background_mask]
            + alpha * frame_blur[background_mask]
        )

    # background.astype(np.uint8),
    return np.stack(plume_masks)


def SVDfiltering(video, k=10):
    import numpy as np

    nframes, height, width = video.shape

    # Reshape to (nframes, height*width)
    video_reshaped = video.reshape(nframes, -1).astype(np.float32)

    # SVD
    U, s, Vt = np.linalg.svd(video_reshaped, full_matrices=False)

    # Reconstruct background with top k components
    S_k = np.diag(s[:k])
    background = U[:, :k] @ S_k @ Vt[:k, :]

    # Subtract background
    foreground = video_reshaped - background

    # Clip to 0-255 and convert back to uint8
    foreground = np.clip(foreground, 0, 255).astype(np.uint8)

    # Reshape back
    video_filtered = foreground.reshape(nframes, height, width)

    return video_filtered

def applyGaussianBlur(video, kernel_size=(5,5)):
    import cv2
    import numpy as np

    nframes = video.shape[0]
    for i in range(nframes):
        frame = video[i]
        frame_blur = cv2.GaussianBlur(frame, kernel_size, 0)
        video[i] = frame_blur

    return video

def triangleThresholding(video, corner_crop_fraction=0.15):
    import cv2
    import numpy as np

    out = video.copy()
    nframes = video.shape[0]

    # Clamp crop ratio to keep a valid central ROI.
    crop = float(np.clip(corner_crop_fraction, 0.0, 0.49))

    for i in range(nframes):
        frame = video[i]

        # Triangle thresholding expects 8-bit single-channel input.
        if frame.dtype != np.uint8:
            frame_for_thresh = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_for_thresh = frame

        h, w = frame_for_thresh.shape[:2]
        y0 = int(h * crop)
        y1 = h - y0
        x0 = int(w * crop)
        x1 = w - x0

        # Compute triangle threshold from center region to avoid dark corner bias.
        roi = frame_for_thresh[y0:y1, x0:x1]
        if roi.size == 0:
            roi = frame_for_thresh

        triangle_value, _ = cv2.threshold(
            roi,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
        )

        _, thresh_frame = cv2.threshold(
            frame_for_thresh,
            triangle_value,
            255,
            cv2.THRESH_BINARY,
        )

        out[i] = thresh_frame

    return out

def applyGlobalThreshold(video, threshold=127):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        _, thresh_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        video[i] = thresh_frame

    return video

def localThreshold(video, blockSize=11, C=2):
    import cv2
    import numpy as np

    nframes = video.shape[0]

    for i in range(nframes):
        frame = video[i]
        thresh_frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, blockSize, C)
        video[i] = thresh_frame

    return video