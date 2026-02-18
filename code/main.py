import argparse
import os
from skimage import io
import numpy as np

from helpers import show_reprojections, get_matches, show_point_cloud, \
    get_markers, show_matches, show_triangulation_topdown
import student


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Homework 3 Camera Geometry")
    parser.add_argument(
        '--sequence',
        required=False,
        default='cards',
        choices=['mikeandikes', 'cards', 'dollar', 'extracredit'],
        help='Which image sequence to use')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where your data is stored')
    parser.add_argument(
        '--ransac-iters',
        type=int,
        default=500,
        help='Number of samples to try in RANSAC')
    parser.add_argument(
        '--num-keypoints',
        type=int,
        default=10000,
        help='Number of keypoints to detect with SIFT')
    parser.add_argument(
        '--no-intermediate-vis',
        action='store_true',
        help='Disables intermediate visualizations'
    )
    parser.add_argument(
        '--visualize-ransac',
        action='store_true',
        help="Visualizes Ransac"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(args.data, args.sequence)
    image_files = sorted(os.listdir(data_dir))

    print(f'Loading {len(image_files)} images for {args.sequence} sequence...')
    images = []
    for image_file in image_files:
        images.append(io.imread(os.path.join(data_dir, image_file)))

    markers = get_markers(os.path.join(args.data, "markers.txt"))

    print('Calculating projection matrices...')
    Ms = []
    for image in images:
        M, residual = student.calculate_projection_matrix(image, markers)
        Ms.append(M)

    if not args.no_intermediate_vis:
        show_reprojections(images, Ms, markers)

    points3d = []
    points3d_color = []

    # Match consecutive pairs (stride 1) and skip-one pairs (stride 2)
    # for denser point cloud coverage
    for stride in [1, 2]:
        for i in range(len(images) - stride):
            image1, M1 = images[i], Ms[i]
            image2, M2 = images[i + stride], Ms[i + stride]

            print(f'Getting matches for images {i + 1} and {i + stride + 1} of {len(images)} (stride {stride})...')
            points1, points2 = get_matches(image1, image2, args.num_keypoints)
            if not args.no_intermediate_vis:
                show_matches(image1, image2, points1, points2)

            print(f'Filtering with RANSAC...')
            F, inliers1, inliers2, residual = student.ransac_fundamental_matrix(
                points1, points2, args.ransac_iters)
            if not args.no_intermediate_vis:
                show_matches(image1, image2, inliers1, inliers2)

            if args.visualize_ransac:
                print(f'Visualizing Ransac')
                student.visualize_ransac()
                student.inlier_counts = []
                student.inlier_residuals = []
                student.final_sampson_distances = []

            print('Calculating 3D points for accepted matches...')
            points3d_found, inliers1_from3d, inliers2_from3d = student.matches_to_3d(
                inliers1, inliers2, M1, M2, 5.0)
            points3d += points3d_found.tolist()
            h, w = image1.shape[:2]
            for point in inliers1_from3d:
                py = np.clip(round(point[1]), 0, h - 1)
                px = np.clip(round(point[0]), 0, w - 1)
                points3d_color.append(tuple(image1[py, px, :] / 255.0))

    for key in markers:
        points3d += markers[key]
        points3d_color += [(0, 0, 0)] * 4

    # Filter points outside the [0, 7]^3 bounding box
    points3d = np.array(points3d)
    points3d_color = np.array(points3d_color)
    mask = np.all((points3d >= 0) & (points3d <= 7), axis=1)
    points3d = points3d[mask]
    points3d_color = points3d_color[mask]

    if not args.no_intermediate_vis:
        show_triangulation_topdown(points3d, points3d_color, Ms)

    show_point_cloud(points3d, points3d_color)


if __name__ == '__main__':
    main()
