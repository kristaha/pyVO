import cv2

from dataloader import DataLoader
from harrisdetector import harris_corners
from pointTracker import PointTracker
from pointProjection import project_points

from debug.PointsVisualizer import PointVisualizer

import time

#dl = DataLoader('dataset/rgbd_dataset_freiburg2_desk') # Edit this string to load a different dataset
dl = DataLoader('dataset/rgbd_dataset_freiburg2_xyz') 

tracker = PointTracker()
vis = PointVisualizer()

# Set initial position of cameras in visualizer
initial_orientation, initial_position = dl.get_transform()
vis.set_groundtruth_transform(initial_orientation, initial_position)
vis.set_estimated_transform(initial_orientation, initial_position)

# Get points for the first frame
grey_img = dl.get_greyscale()
depth_img = dl.get_depth()
points_and_response = harris_corners(grey_img)
print("Harris corners found: " + str(points_and_response))
tracker.add_new_corners(grey_img, points_and_response)

# Project the points in the first frame
previous_ids, previous_points = tracker.get_position_with_id()
previous_ids, previous_points = project_points(previous_ids, previous_points, depth_img)
vis.set_projected_points(previous_points, initial_orientation, initial_position)

current_orientation = initial_orientation
current_position = initial_position

iterations = 0
start_time = time.time()

while dl.has_next():
    dl.next()

    iterations += 1

    if(iterations % 30 == 0):
        fps = iterations / (time.time() - start_time)
        print(f"Average frames per second: {fps}")


    # Visualization
    gt_position, gt_orientation = dl.get_transform()
    vis.set_groundtruth_transform(gt_position, gt_orientation)

    # Get images
    grey_img = dl.get_greyscale()
    depth_img = dl.get_depth()

    # Track current points on new image
    tracker.track_on_image(grey_img)
    tracker.visualize(grey_img)

    # Project tracked points
    ids, points = tracker.get_position_with_id()
    if (ids.size > 0)  and (points.size > 0 ):
        ids, points = project_points(ids, points, depth_img)
        vis.set_projected_points(points, gt_position, gt_orientation)

    # Replace lost points
    points_and_response = harris_corners(grey_img)
    tracker.add_new_corners(grey_img, points_and_response)


cv2.destroyAllWindows()
