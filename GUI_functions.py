def set_spray_origin(file, rotated_video, firstFrameNumber, nframes, height):    
    import cv2
    import json    
    import os

    # Load saved spray origins
    origins_file = 'spray_origins.json'
    if os.path.exists(origins_file):
        with open(origins_file, 'r') as f:
            spray_origins = json.load(f)
    else:
        spray_origins = {}

    # Set spray origin
    if file in spray_origins:
        spray_origin = tuple(spray_origins[file])
        print(f"Reusing spray origin for {file}: {spray_origin}")
    else:
        # UI to select
        class PointHolder:
            def __init__(self):
                self.point = None
        
        holder = PointHolder()
        def select_origin(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                holder.point = (x, y)  # type: ignore
                print(f"Selected spray origin: {holder.point}")
        
        cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[firstFrameNumber+100]) # Show a frame after firstFrameNumber for context, may need adjustment
        cv2.setMouseCallback('Set Spray Origin - Click on the nozzle', select_origin)
        
        current_frame = firstFrameNumber + 100
        while holder.point is None:
            key = cv2.waitKeyEx(10)
            if key == ord('q'):
                break
            elif key == 2424832:  # left arrow
                current_frame = max(firstFrameNumber, current_frame - 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
            elif key == 2555904:  # right arrow
                current_frame = min(nframes - 1, current_frame + 1)
                cv2.imshow('Set Spray Origin - Click on the nozzle', rotated_video[current_frame])
        cv2.destroyWindow('Set Spray Origin - Click on the nozzle')
        
        if holder.point is None:
            spray_origin = (1, height // 2)  # Default
        else:
            spray_origin = holder.point
        
        # Save
        spray_origins[file] = list(spray_origin)
        with open(origins_file, 'w') as f:
            json.dump(spray_origins, f)

        print(f"Spray origin for {file}: {spray_origin}")

    return spray_origin

def draw_freehand_mask(video_strip):
    import cv2
    import numpy as np

    nframes, height, width = video_strip.shape[:3]
    
    drawing = False
    points = []

    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, points, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            points.append((x, y))
            cv2.line(mask, points[-2], points[-1], 255, thickness=2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

            if len(points) > 2:
                contour = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [contour], 255)

            points = []

    frame = video_strip[nframes // 2]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Draw Background Mask")
    cv2.setMouseCallback("Draw Background Mask", draw_mask)

    while True:
        # Ensure overlay is 3-channel BGR (frame may be grayscale)
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            overlay = frame.copy()

        # Apply red overlay safely (works even if mask has no 255 pixels)
        mask_bool3 = (mask == 255)[:, :, None]
        overlay = np.where(mask_bool3, np.array([0, 0, 255], dtype=overlay.dtype), overlay)

        cv2.imshow("Draw Background Mask", overlay)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # reset mask
            mask[:] = 0

    cv2.destroyAllWindows()
    cv2.imwrite("background_mask.png", mask)


def draw_and_compare_mask_frames(original_video, predicted_masks, start_frame=0, save_prefix="validation"):
    import cv2
    import numpy as np

    drawing = False
    points = []

    nframes = min(len(original_video), len(predicted_masks))
    if nframes == 0:
        raise ValueError("No frames available for validation")

    frame_index = int(np.clip(start_frame, 0, nframes - 1))
    h, w = predicted_masks.shape[1:3]
    gt_masks = np.zeros((nframes, h, w), dtype=np.uint8)

    window_name = "Draw Validation Mask"

    def get_mask_outline(mask_uint8, thickness=2):
        mask_bin = (mask_uint8 > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = np.zeros_like(mask_bin, dtype=np.uint8)
        if len(contours) > 0:
            cv2.drawContours(outline, contours, -1, 255, thickness=thickness)
        return outline

    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, points, gt_masks, frame_index

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            points.append((x, y))
            cv2.line(gt_masks[frame_index], points[-2], points[-1], 255, thickness=2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(points) > 2:
                contour = np.array(points, dtype=np.int32)
                cv2.fillPoly(gt_masks[frame_index], [contour], 255)
            points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_mask)

    while True:
        current_frame = original_video[frame_index]
        current_pred = predicted_masks[frame_index]
        current_gt = gt_masks[frame_index]

        if current_frame.ndim == 2 or (current_frame.ndim == 3 and current_frame.shape[2] == 1):
            overlay = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
        else:
            overlay = current_frame.copy()

        if overlay.shape[:2] != current_pred.shape[:2]:
            current_pred = cv2.resize(current_pred, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
        if overlay.shape[:2] != current_gt.shape[:2]:
            current_gt = cv2.resize(current_gt, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_bin = (current_pred > 0)
        gt_bin = (current_gt > 0)
        pred_outline = get_mask_outline(current_pred, thickness=2) > 0

        # Predicted final mask shown as 50% transparent blue outline.
        pred_layer = overlay.copy()
        pred_layer[pred_outline] = [255, 0, 0]
        overlay = cv2.addWeighted(pred_layer, 0.5, overlay, 0.5, 0.0)

        # Drawn mask in red, and overlap with predicted outline in green.
        overlay[gt_bin] = [0, 0, 255]
        overlay[pred_outline & gt_bin] = [0, 255, 0]

        cv2.putText(overlay, f"Frame {frame_index}/{nframes - 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, "Draw GT | Left/Right: frame | q: finish | r: reset frame", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window_name, overlay)

        key = cv2.waitKeyEx(40)
        if key == ord('q'):
            break
        if key == ord('r'):
            gt_masks[frame_index][:] = 0
        elif key == 2424832:  # left arrow
            frame_index = max(0, frame_index - 1)
            points = []
            drawing = False
        elif key == 2555904:  # right arrow
            frame_index = min(nframes - 1, frame_index + 1)
            points = []
            drawing = False

    cv2.destroyWindow(window_name)

    final_gt_mask = gt_masks[frame_index]
    final_pred_mask = predicted_masks[frame_index]
    if final_pred_mask.shape != final_gt_mask.shape:
        final_pred_mask = cv2.resize(final_pred_mask, (final_gt_mask.shape[1], final_gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    gt_bin = (final_gt_mask > 0).astype(np.uint8)
    pred_bin = (final_pred_mask > 0).astype(np.uint8)

    intersection = int(np.logical_and(gt_bin == 1, pred_bin == 1).sum())
    gt_area = int(gt_bin.sum())
    pred_area = int(pred_bin.sum())
    union = int(np.logical_or(gt_bin == 1, pred_bin == 1).sum())
    total = int(gt_bin.size)

    iou = (intersection / union) if union > 0 else 0.0
    dice = (2.0 * intersection / (gt_area + pred_area)) if (gt_area + pred_area) > 0 else 0.0
    precision = (intersection / pred_area) if pred_area > 0 else 0.0
    recall = (intersection / gt_area) if gt_area > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    pixel_accuracy = float((gt_bin == pred_bin).sum()) / float(total)

    metrics = {
        "frame_index": int(frame_index),
        "intersection": intersection,
        #"union": union,
        "gt_area": gt_area,
        "pred_area": pred_area,
        #"iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        #"f1": float(f1),
        "pixel_accuracy": float(pixel_accuracy),
    }

    cv2.imwrite(f"{save_prefix}_gt_mask_frame_{frame_index}.png", final_gt_mask)

    if original_video[frame_index].ndim == 2 or (original_video[frame_index].ndim == 3 and original_video[frame_index].shape[2] == 1):
        comparison = cv2.cvtColor(original_video[frame_index], cv2.COLOR_GRAY2BGR)
    else:
        comparison = original_video[frame_index].copy()

    pred_outline_final = get_mask_outline(final_pred_mask, thickness=2) > 0
    gt_outline_final = get_mask_outline(final_gt_mask, thickness=2) > 0
    comp_pred_layer = comparison.copy()
    comp_pred_layer[pred_outline_final] = [255, 0, 0]
    comparison = cv2.addWeighted(comp_pred_layer, 0.5, comparison, 0.5, 0.0)

    comparison[gt_outline_final] = [0, 0, 255]
    comparison[np.logical_and(pred_outline_final, gt_outline_final)] = [0, 255, 0]
    cv2.imwrite(f"{save_prefix}_comparison_frame_{frame_index}.png", comparison)

    return metrics, final_gt_mask