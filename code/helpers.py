import cv2
import numpy as np
from skimage import img_as_float32
import matplotlib.pyplot as plt
import random


def get_markers(markers_path):
    """
    Returns a dictionary mapping a marker ID to a 4x3 array
    containing the 3d points for each of the 4 corners of the
    marker in our scanning setup
    """
    markers = {}
    with open(markers_path) as f:
        first_dim = 0
        second_dim = 0
        for i, line in enumerate(f.readlines()):
            if i == 0:
                first_dim, second_dim = [float(x) for x in line.split()]
            else:
                info = [float(x) for x in line.split()]
                markers[i] = [
                    [info[0], info[1], info[2]],
                    [info[0] + first_dim * info[3], info[1] + first_dim * info[4], info[2] + first_dim * info[5]],
                    [info[0] + first_dim * info[3] + second_dim * info[6], info[1] + first_dim * info[4] + second_dim * info[7], info[2] + first_dim * info[5] + second_dim * info[8]],
                    [info[0] + second_dim * info[6], info[1] + second_dim * info[7], info[2] + second_dim * info[8]],
                ]
    return markers


def get_matches(image1, image2, num_keypoints=5000):
    """
    Wraps OpenCV's SIFT function and feature matcher.
    Returns two N x 2 numpy arrays, 2d points in image1 and image2
    that are proposed matches.
    """
    # Find keypoints and descriptors with SIFT
    sift = cv2.SIFT_create(nfeatures=num_keypoints)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Match descriptors using 2NN + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extract matched keypoints
    points1 = np.array([kp1[m.queryIdx].pt for m in good])
    points2 = np.array([kp2[m.trainIdx].pt for m in good])

    return points1, points2


def show_matches(image1, image2, points1, points2):
    """
    Shows matches from image1 to image2, represented by Nx2 arrays
    points1 and points2
    """
    image1 = img_as_float32(image1)
    image2 = img_as_float32(image2)

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Matches between image pair.")
    plt.axis('off')

    matches_image = np.hstack([image1, image2])
    plt.imshow(matches_image)

    shift = image1.shape[1]
    for i in range(0, points1.shape[0]):

        random_color = lambda: random.randint(0, 255)
        cur_color = ('#%02X%02X%02X' % (random_color(), random_color(), random_color()))

        x1 = points1[i, 1]
        y1 = points1[i, 0]
        x2 = points2[i, 1]
        y2 = points2[i, 0]

        x = np.array([x1, x2])
        y = np.array([y1, y2 + shift])
        plt.plot(y, x, c=cur_color, linewidth=0.5)

    plt.show()


def reproject_points(M, points):
    """
    Use projection matrix to project Nx3 array of 3d points into Nx2
    array of image points
    """

    reshaped_points = np.concatenate(
        (points, np.ones((points.shape[0], 1))), axis=1)
    projected_points = np.matmul(M, np.transpose(reshaped_points))
    projected_points = np.transpose(projected_points)
    u = np.divide(projected_points[:, 0], projected_points[:, 2])
    v = np.divide(projected_points[:, 1], projected_points[:, 2])
    projected_points = np.transpose(np.vstack([u, v]))

    return projected_points


def show_reprojections(images, Ms, markers):
    """
    Show reprojected markers in each image
    """
    points3d = []

    for marker_id in markers:
        points3d += markers[marker_id]
    points3d = np.array(points3d)

    fig, axs = plt.subplots(1, len(images), figsize=(15, 6))
    fig.canvas.manager.set_window_title("Reprojected markers for each image.")
    plt.axis('off')

    for i in range(len(images)):
        points2d = reproject_points(Ms[i], points3d)
        axs[i].imshow(images[i])
        axs[i].scatter(points2d[:, 0], points2d[:, 1])
    plt.show()


def show_triangulation_topdown(points3d, points3d_color, Ms,
                                reproj_errors=None, rejected_points=None):
    """Top-down (bird's eye) view of triangulated 3D points and cameras.

    Panel 1: XZ plane (top-down)
    Panel 2: XY plane (front view)
    Panel 3: Reprojection error histogram (if errors provided)
    """
    n_panels = 3 if reproj_errors is not None and len(reproj_errors) > 0 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.canvas.manager.set_window_title("Triangulation: front and top-down views")

    # Extract camera centers from projection matrices: C = null space of M
    cam_centers = []
    for M in Ms:
        # C = -M[:,:3]^{-1} @ M[:,3]
        try:
            C = -np.linalg.inv(M[:, :3]) @ M[:, 3]
            cam_centers.append(C)
        except np.linalg.LinAlgError:
            pass

    # Color by reprojection error if available, otherwise use point colors
    if reproj_errors is not None and len(reproj_errors) > 0:
        errs = np.array(reproj_errors)
        # Normalize errors for colormap: green=low, red=high
        vmax = min(np.percentile(errs, 95), 10.0)
        colors_mapped = plt.cm.RdYlGn_r(np.clip(errs / max(vmax, 1e-6), 0, 1))
    else:
        colors_mapped = points3d_color

    # Panel 1: XZ plane (top-down)
    ax1 = axes[0]
    ax1.scatter(points3d[:, 0], points3d[:, 2], c=colors_mapped, s=1, alpha=0.5)
    if rejected_points is not None and len(rejected_points) > 0:
        rej = np.array(rejected_points)
        ax1.scatter(rej[:, 0], rej[:, 2], c='gray', marker='x', s=8, alpha=0.3, label='Rejected')
        ax1.legend(fontsize=8)
    for i, C in enumerate(cam_centers):
        ax1.plot(C[0], C[2], 'k^', markersize=10)
        ax1.annotate(f'C{i+1}', (C[0], C[2]), textcoords='offset points',
                     xytext=(5, 5), fontsize=8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Front (XZ)')
    ax1.set_aspect('equal')

    # Panel 2: XY plane (top-down view)
    ax2 = axes[1]
    ax2.scatter(points3d[:, 0], points3d[:, 1], c=colors_mapped, s=1, alpha=0.5)
    if rejected_points is not None and len(rejected_points) > 0:
        ax2.scatter(rej[:, 0], rej[:, 1], c='gray', marker='x', s=8, alpha=0.3)
    for i, C in enumerate(cam_centers):
        ax2.plot(C[0], C[1], 'k^', markersize=10)
        ax2.annotate(f'C{i+1}', (C[0], C[1]), textcoords='offset points',
                     xytext=(5, 5), fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-down (XY)')
    ax2.set_aspect('equal')

    # Panel 3: Reprojection error histogram
    if n_panels == 3:
        ax3 = axes[2]
        ax3.hist(reproj_errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax3.axvline(x=5.0, color='red', linestyle='--', label='Threshold (5.0 px)')
        ax3.set_xlabel('Reprojection error (px)')
        ax3.set_ylabel('Count')
        ax3.set_title('Reprojection error distribution')
        ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def show_point_cloud(points3d, colors):
    """
    Show 3D points with their corresponding colors.
    Marker size adapts to point count for readable visualizations.
    """
    n = len(points3d)
    # Scale marker size inversely with point count
    if n > 10000:
        s = 1.0
    elif n > 3000:
        s = 1.0
    else:
        s = 4.0

    fig = plt.figure(figsize=(9, 9))
    fig.canvas.manager.set_window_title("Recovered 3D points.")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
               c=colors, s=s, alpha=0.8, edgecolors='none')

    # Equal aspect ratio for all axes
    mid = points3d.mean(axis=0)
    span = (points3d.max(axis=0) - points3d.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Reconstructed 3D points ({n:,})")

    # Cleaner background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()
    plt.show()