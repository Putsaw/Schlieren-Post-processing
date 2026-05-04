import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def compute_frame_histogram(frame, bins=256, mask=None):
    """
    Compute histogram for a single frame.
    
    Args:
        frame: Input frame (grayscale or BGR)
        bins: Number of histogram bins
        mask: Optional mask to compute histogram only for certain regions
    
    Returns:
        histogram: Normalized histogram values
    """
    if frame.ndim == 3:  # BGR image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([frame], [0], mask, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def plot_histogram_change_heatmap(video_strip, firstFrameNumber, bin_resolution=64):
    """
    Show histogram evolution as a 2D heatmap (time vs intensity bins).
    Useful for seeing how the overall intensity distribution changes over time.
    
    Args:
        video_strip: Array of video frames
        firstFrameNumber: First frame to analyze
        bin_resolution: Number of bins to downsample to (for visualization)
    """
    nframes = video_strip.shape[0]
    histogram_matrix = np.zeros((nframes - firstFrameNumber, bin_resolution))
    
    for idx in range(firstFrameNumber, nframes):
        hist = compute_frame_histogram(video_strip[idx], bins=bin_resolution)
        histogram_matrix[idx - firstFrameNumber] = hist
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(histogram_matrix.T, aspect='auto', origin='lower', 
                    cmap='viridis', interpolation='nearest')
    ax.set_xlabel("Frame Number (relative to first frame)")
    ax.set_ylabel("Intensity Bin")
    ax.set_title("Histogram Evolution Over Time")
    plt.colorbar(im, ax=ax, label="Normalized Count")
    plt.tight_layout()
    plt.show()

def analyze_histogram_statistics(video_strip, firstFrameNumber):
    """
    Compute and plot summary statistics of histogram changes over time.
    
    Args:
        video_strip: Array of video frames
        firstFrameNumber: First frame to analyze
    
    Returns:
        Dictionary with statistics
    """
    nframes = video_strip.shape[0]
    
    mean_intensities = []
    std_intensities = []
    entropy_values = []
    
    for idx in range(firstFrameNumber, nframes):
        frame = video_strip[idx]
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mean_intensities.append(np.mean(frame))
        std_intensities.append(np.std(frame))
        
        # Compute entropy (measure of histogram spread)
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        entropy_values.append(entropy)
    
    frames = np.arange(firstFrameNumber, nframes)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    
    axes[0].plot(frames, mean_intensities)
    axes[0].set_title("Mean Intensity Over Time")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(frames, std_intensities)
    axes[1].set_title("Intensity Std Dev Over Time")
    axes[1].set_ylabel("Std Dev")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(frames, entropy_values)
    axes[2].set_title("Histogram Entropy Over Time")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel("Frame Number")
    
    fig.tight_layout()
    plt.show()
    
    return {
        'mean_intensities': np.array(mean_intensities),
        'std_intensities': np.array(std_intensities),
        'entropy_values': np.array(entropy_values)
    }

def draw_single_frame_histogram(frame, hist_width=512, hist_height=400, frame_number=None):
    """
    Create a histogram image for a single frame that can be displayed alongside video.
    
    Args:
        frame: Input frame (grayscale or BGR)
        hist_width: Width of histogram image in pixels
        hist_height: Height of histogram image in pixels
        frame_number: Optional frame number to display
    
    Returns:
        hist_image: Histogram visualization as a BGR image (h, w, 3) suitable for cv2.imshow/cv2.imwrite
    """
    # Compute histogram
    hist = compute_frame_histogram(frame)
    
    # Create histogram visualization with margins for axes and labels
    margin_left = 50
    margin_bottom = 40
    margin_top = 20
    plot_width = hist_width - margin_left - 20
    plot_height = hist_height - margin_bottom - margin_top
    
    hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    hist_image[:] = (200, 200, 200)  # Light gray background
    
    # Draw axes
    axis_color = (0, 0, 0)
    # Y-axis (left)
    cv2.line(hist_image, (margin_left, margin_top), (margin_left, hist_height - margin_bottom), axis_color, 2)
    # X-axis (bottom)
    cv2.line(hist_image, (margin_left, hist_height - margin_bottom), (hist_width - 20, hist_height - margin_bottom), axis_color, 2)
    
    # Draw histogram bars
    max_val = np.max(hist) * 1.1 if np.max(hist) > 0 else 1
    bin_width = plot_width // 256
    
    for i in range(256):
        bar_height = int(hist[i] / max_val * plot_height)
        x = margin_left + i * bin_width
        y_top = hist_height - margin_bottom - bar_height
        cv2.rectangle(hist_image, (x, y_top), 
                     (x + bin_width, hist_height - margin_bottom), (0, 255, 0), -1)
    
    # Add axis labels
    label_color = (0, 0, 0)
    
    # X-axis label
    cv2.putText(hist_image, "Intensity", (hist_width // 2 - 40, hist_height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 1)
    
    # Y-axis label (rotated, so we'll place it vertically)
    cv2.putText(hist_image, "Count", (5, hist_height // 2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 1)
    
    # Add tick labels on x-axis (0, 64, 128, 192, 255)
    tick_positions = [0, 64, 128, 192, 255]
    for tick in tick_positions:
        x_pos = margin_left + int(tick * bin_width)
        cv2.line(hist_image, (x_pos, hist_height - margin_bottom), 
                (x_pos, hist_height - margin_bottom + 5), axis_color, 1)
        cv2.putText(hist_image, str(tick), (x_pos - 10, hist_height - margin_bottom + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    
    # Add frame stats
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame
    
    mean_intensity = np.mean(frame_gray)
    std_intensity = np.std(frame_gray)
    
    text_color = (0, 0, 0)
    cv2.putText(hist_image, "Histogram", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    if frame_number is not None:
        cv2.putText(hist_image, f"Frame: {frame_number}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1)
    cv2.putText(hist_image, f"Mean: {mean_intensity:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1)
    cv2.putText(hist_image, f"Std: {std_intensity:.1f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1)
    
    return hist_image


def plot_frame_histogram(frame, frame_number=None, bins=256):
    """
    Plot a histogram for a single frame using matplotlib.

    Args:
        frame: Input frame (grayscale or BGR)
        frame_number: Optional frame number to display in the title
        bins: Number of histogram bins
    """
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    pixels = frame_gray.flatten()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pixels, bins=bins, range=(0, 256), color='green', edgecolor='none')
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    title = "Frame Histogram" if frame_number is None else f"Frame {frame_number} Histogram"
    ax.set_title(title)
    ax.set_xlim([0, 255])
    ax.grid(True, alpha=0.3)

    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.legend(title=f"Std: {std_val:.1f}")

    fig.tight_layout()
    plt.show()


def plot_fft_frequency_image(frame, frame_number=None, bins=256, use_log_magnitude=True):
    """
    Plot the 2D FFT magnitude spectrum image for a single frame.

    Args:
        frame: Input frame (grayscale or BGR)
        frame_number: Optional frame number to display in the title
        bins: Kept for backward compatibility (unused)
        use_log_magnitude: If True, display log(1 + |FFT|)
    """
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    frame_f = frame_gray.astype(np.float32)

    # 2D FFT -> shift DC to center -> magnitude spectrum
    fft2 = np.fft.fft2(frame_f)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shift)

    if use_log_magnitude:
        spectrum = np.log1p(magnitude)
        title_base = "FFT 2D Magnitude Spectrum (log scale)"
    else:
        spectrum = magnitude
        title_base = "FFT 2D Magnitude Spectrum"

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(spectrum, cmap='magma', origin='lower')
    title = title_base if frame_number is None else f"{title_base} - Frame {frame_number}"
    ax.set_title(title)
    ax.set_xlabel("Frequency u")
    ax.set_ylabel("Frequency v")
    plt.colorbar(im, ax=ax, label="Magnitude")

    fig.tight_layout()
    plt.show()


def render_histogram_to_array(frame, frame_number=None, bins=256, fig_width=6, fig_height=4):
    """
    Render a matplotlib histogram for a single frame into a BGR numpy array
    suitable for side-by-side display with an OpenCV window.

    Args:
        frame: Input frame (grayscale or BGR)
        frame_number: Optional frame number to include in the title
        bins: Number of histogram bins
        fig_width: Matplotlib figure width in inches
        fig_height: Matplotlib figure height in inches

    Returns:
        hist_bgr: BGR numpy array of the rendered histogram
    """
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    pixels = frame_gray.flatten()
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.hist(pixels, bins=bins, range=(0, 256), color='green', edgecolor='none')
    ax.set_xlabel("Intensity (8 bit, 0-255)")
    ax.set_ylabel("Count of Pixels")
    title = "Frame Histogram" if frame_number is None else f"Frame {frame_number} Histogram"
    ax.set_title(title)
    ax.set_xlim([0, 255])
    ax.grid(True, alpha=0.3)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.legend(title=f"Std: {std_val:.1f}")
    fig.tight_layout()

    # Render figure to RGBA buffer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    hist_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return hist_bgr


def display_histogram_animation(video_strip, firstFrameNumber, last_frame=None, delay=50):
    """
    Display histograms frame by frame with OpenCV (interactive).
    
    Args:
        video_strip: Array of video frames
        firstFrameNumber: First frame to analyze
        last_frame: Last frame to process (None = all frames)
        delay: Delay in ms between frames (press 'q' to quit, 'p' to pause)
    """
    if last_frame is None:
        last_frame = video_strip.shape[0]
    
    for idx in range(firstFrameNumber, last_frame):
        frame = video_strip[idx]
        hist_image = draw_single_frame_histogram(frame, frame_number=idx)
        
        cv2.imshow('Histogram Evolution', hist_image)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
    
    cv2.destroyAllWindows()

    