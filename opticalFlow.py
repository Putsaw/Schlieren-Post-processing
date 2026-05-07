def opticalFlowFarnebackCalculation(prev_frame, frame):
    import cv2

    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, # type: ignore
                                        0.5,  # pyramid scale
                                        3,    # levels
                                        15,   # window size
                                        3,    # iterations
                                        5,    # poly_n
                                        1.2,  # poly_sigma
                                        0)    # type: ignore # flags

    return flow

def opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow):
    import cv2
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        gray = frame.copy()

    flow = deepflow.calc(prev_gray, gray, None) # type: ignore

    return flow

def runOpticalFlowCalculation(firstFrameNumber, video, method, deepflow=None):
    import cv2
    import numpy as np
    from clustering import create_cluster_mask, overlay_cluster_outline 

    first_frame = video[firstFrameNumber]
    prev_frame = first_frame

    nframes = video.shape[0]
    cluster_masks = np.zeros_like(video, dtype=np.uint8)
    clustered_overlays =  np.zeros_like(video, dtype=np.uint8)
    masks =  np.zeros_like(video, dtype=np.uint8)

    if method == 'Farneback':
        for i in range(firstFrameNumber, nframes):
            frame = video[i]

            # --- Compute DeepFlow optical flow ---
            flow = opticalFlowFarnebackCalculation(prev_frame, frame) # Farneback 0.3 threshold

            # Compute magnitude (motion strength) and angle (not needed here)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # threshold movement
            mask = (mag > 0.4).astype(np.uint8) * 255

            # clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

            # Use clustering to get signal outlines
            cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)

            clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

            prev_frame = frame

            cluster_masks[i] = cluster_mask
            clustered_overlays[i] = clustered_overlay # only for visualization
            masks[i] = mask # only for visualization
            print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks

    # Not fully implemented yet, but structure is in place
    elif method == 'DeepFlow':
        if deepflow is None:
            raise ValueError("DeepFlow instance must be provided for DeepFlow method.")
        for i in range(firstFrameNumber, nframes):
            frame = video[i]

            # --- Compute DeepFlow optical flow ---
            flow = opticalFlowDeepFlowCalculation(prev_frame, frame, deepflow) # DeepFlow 1.0 threshold

            # Compute magnitude (motion strength) and angle (not needed here)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # threshold movement
            mask = (mag > 1.0).astype(np.uint8) * 255

            # clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

            # Use clustering to get signal outlines
            cluster_mask = create_cluster_mask(mask, cluster_distance=50, alpha=40)
            clustered_overlay = overlay_cluster_outline(frame, cluster_mask)

            prev_frame = frame

            cluster_masks[i] = cluster_mask
            clustered_overlays[i] = clustered_overlay # only for visualization
            masks[i] = mask # only for visualization
            print(f"Processed frame {i+1}/{nframes}")

        return cluster_masks, clustered_overlays, masks
    else:
        raise ValueError(f"Unsupported optical flow method: {method}")
    


def runOpticalFlowCalculationWeighted(firstFrameNumber, video, method, deepflow=None, workers=None):
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    first_frame = video[firstFrameNumber]
    nframes = video.shape[0]
    mag_array = np.zeros_like(video, dtype=np.float32)

    if method != 'Farneback':
        raise ValueError(f"Unsupported optical flow method: {method}")

    indices = list(range(firstFrameNumber, nframes))
    if workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(max(1, cpu), len(indices))
    else:
        max_workers = max(1, min(int(workers), len(indices)))

    # sequential fallback
    if max_workers <= 1:
        prev_frame = first_frame
        for i in indices:
            frame = video[i]
            flow = opticalFlowFarnebackCalculation(prev_frame, frame)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_array[i] = mag
            prev_frame = frame
            print(f"Processed frame {i+1}/{nframes}")
        return mag_array

    def _worker_task(i, prev_frame, frame):
        flow = opticalFlowFarnebackCalculation(prev_frame, frame)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return i, mag

    tasks = []
    for i in indices:
        prev = first_frame if i == firstFrameNumber else video[i - 1]
        frame = video[i]
        tasks.append((i, prev, frame))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_task, t[0], t[1], t[2]): t[0] for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                i, mag = fut.result()
            except Exception as e:
                raise RuntimeError(f"Optical flow worker failed for frame {idx}: {e}")
            mag_array[i] = mag
            completed += 1
            if (completed % 10 == 0) or (completed == len(futures)):
                print(f"Processed {completed}/{len(futures)} optical flow tasks")

    return mag_array