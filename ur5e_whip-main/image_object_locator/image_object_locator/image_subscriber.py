from collections import deque
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import cv2 as cv
import numpy as np
import rclpy

from image_object_locator import imgproc

resource = "src/ur5e_whip/image_object_locator/resource/"
camera_calibration_document = resource + "camera_calibration.txt"

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # store camera calibration matrix
        # [ fx  0   cx ]
        # [ 0   fy  cy ]
        # [ 0   0   1  ]
        # fx, fy are the focal lengths in pixels
        # cx, cy are the coordinates of the principal point 
        # self.A = np.array([[1.22109334e+03, 0.00000000e+00, 6.56137838e+02],
                        #    [0.00000000e+00, 1.22328544e+03, 4.85644100e+02],
                        #    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # and distortion coefficients
        # [ k1, k2, p1, p2, k3 ]
        # k1, k2 are the radial distortion coefficients
        # p1, p2 are the tangential distortion coefficients
        # k3 is the third radial distortion coefficient
        # self.dist_coeffs = np.array([1.88988104e-01, -6.15174566e-01, -9.88286870e-05, 2.23374393e-04, 5.87519850e-01])
        
        self.A, self.dist_coeffs = self.parse_calibration_document(camera_calibration_document)
        
        # self.path_image_anno = resource + "annotated_pingpong_ball_5.png"
        # self.path_image_orig = resource + "original_pingpong_ball_5.png"
        
        # # original image in rgb, and lab color spaces
        # self.orig_img = imgproc.load_image(self.path_image_orig, color_space=['rgb'])
        # self.orig_img_rgb = self.orig_img[0]
        # # annotated image in rgb, and lab color spaces
        # self.anno_img = imgproc.load_image(self.path_image_anno, color_space=['rgb'])
        # self.anno_img_rgb = self.anno_img[0]
        
        # self.anno_color_rgb = np.array([0, 0, 255], dtype=np.uint8)
        
        # # Get annotated color data
        # self.pingpong_color_data_rgb = imgproc.get_annotated_color_data(self.orig_img_rgb, self.anno_img_rgb, self.anno_color_rgb)
        
        self.path_anno_folder = resource + "/annotations/"
        self.pingpong_color_data_list_rgb = imgproc.process_annotated_images(self.path_anno_folder)
        # Extract means and stds
        means = [data[2] for data in self.pingpong_color_data_list_rgb]  # Means of all images
        stds = [data[3] for data in self.pingpong_color_data_list_rgb]  # Stds of all images

        # Compute the mean and std across the batch
        self.pingpong_color_data_rgb = [None, None, np.mean(means, axis=0), np.std(stds, axis=0)]

        # Print out the RGB and Lab means and standard deviations
        self.get_logger().info("Pingpong color (RGB):\nMean: %s\nStd: %s" % (self.pingpong_color_data_rgb[2], self.pingpong_color_data_rgb[3]))
        
        # plot_color_distribution(self.pingpong_color_data_rgb[1], 'RGB', False)

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        self.pos_pub = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        self.transform_pub = self.create_publisher(
            Float64MultiArray,
            '/target_reference_frame',
            10)
        
        self.rgb_sub  # prevent unused variable warning
        self.depth_sub  # prevent unused variable warning
        self.br = CvBridge()
        
        self.img_bgr, self.img_bgr_undist, self.img_rgb, self.img_rgb_undist, self.img_depth = None, None, None, None, None
        
        # Initialize the sliding window for (x, y) coordinates
        self.window_size = 5
        self.xy_window = deque(maxlen=self.window_size)
        self.xy_predicted = None
        self.tolerance = 10 
    
    
    def parse_calibration_document(self, file_path):
        """
        Parse the camera calibration document to extract the camera matrix and distortion coefficients.
        The document is expected to contain the camera matrix and distortion coefficients in a specific format.
        
        Returns:
            camera_matrix (np.ndarray): The camera matrix.
            distortion_params (np.ndarray): The distortion coefficients.
        """
        with open(file_path, 'r') as file:
            doc = file.read()

        lines = [line.strip() for line in doc.strip().split("\n") if line.strip() and not line.startswith("Calibration matrix:") and not line.startswith("Distortion parameters")]

        camera_matrix_str = lines[0] + "\n" + lines[1] + "\n" + lines[2]
        camera_matrix_str = camera_matrix_str.strip('[]')
        camera_matrix_str = camera_matrix_str.replace("][", "\n")

        camera_matrix_str = camera_matrix_str.replace('] ', ' ')
        camera_matrix_str = camera_matrix_str.replace('[', '').replace(']', '')

        camera_matrix = np.array([[float(num) for num in row.split()] for row in camera_matrix_str.split("\n")])

        distortion_params_str = lines[3].strip("[]")
        distortion_params = np.array([float(num) for num in distortion_params_str.split()])
        
        return camera_matrix, distortion_params
    
    
    def predict_next_xy(self):
        """
        Predict the next (x, y) position based on the current sliding window.
        A simple linear prediction based on the average change in x, y can be applied.
        
        Returns:
            tuple: Predicted (x, y) coordinates.
        """
        if len(self.xy_window) < 2:
            return self.xy_window[-1]  # Return the last known position if there's not enough data
        
        # Calculate the average change in x and y over the window
        dx = np.mean([self.xy_window[i + 1][0] - self.xy_window[i][0] for i in range(len(self.xy_window) - 1)])
        dy = np.mean([self.xy_window[i + 1][1] - self.xy_window[i][1] for i in range(len(self.xy_window) - 1)])

        # Predict the next position based on the average change
        last_x, last_y = self.xy_window[-1]
        pred_x = last_x + dx
        pred_y = last_y + dy

        return int(pred_x), int(pred_y)
    

    def pixel_to_camera(self, m, A, Z, T):
        """
        Convert pixel coordinates to camera frame coordinates.
            i.e. s*m' = A[R|t]M'
            or
                [u    [fx  0   cx    [r11 r12 r13 t1    [X
                s  v  =  0   fy  cy  =  r21 r22 r23 t2  *  Y
                1]    0   0   1 ]    r31 r32 r33 t3]    Z
                                                        1]
            We know m, u, v, A, and Z
            We want to find X and Y
        
        Args:
            m (tuple): Pixel coordinates (u, v).
            A (np.ndarray): Camera intrinsic matrix.
            Z (float): Depth value in meters.
            T (np.ndarray): Camera extrinsic matrix (rotation and translation).
            
        Returns:
            tuple: Camera frame coordinates (X, Y, Z).
        """
        u = m[0]
        v = m[1]
        fx = A[0, 0]
        fy = A[1, 1]
        cx = A[0, 2]
        cy = A[1, 2]
        
        R = T[:3, :3]
        t = T[:3, 3]
        
        X = -((cx * t[2] + fx * t[0] + (v - cy) * (cx * R[2, 1] + fx * R[0, 1]) + Z * (cx * R[2, 2] + fx * R[0, 2]) - u * (t[2] + (v - cy) * R[2, 1] + Z * R[2, 2])) / (cx * R[2, 0] + fx * R[0, 0] - R[2, 0] * u))
        Y = -((cy * t[2] + fy * t[1] + (u - cx) * (cy * R[2, 0] + fy * R[1, 0]) + Z * (cy * R[2, 2] + fy * R[1, 2]) - v * (t[2] + (u - cx) * R[2, 0] + Z * R[2, 2])) / (cy * R[2, 1] + fy * R[1, 1] - R[2, 1] * v))
        
        # Flip sign of X and Y to match the camera frame.
        return -X, -Y, Z
    
    def process_or_not(self):
        '''
        Check if both images are available and call the listener callback.
        '''
        if(self.img_rgb_undist is not None and self.img_depth is not None):
            self.listener_callback()
            self.img_rgb_undist = None
            self.img_depth = None
    
    def rgb_callback(self, msg):
        '''
        Callback function for the RGB image subscriber.
        Converts the ROS image message to OpenCV format and undistorts it.
        '''
        # self.img_bgr = cv.flip(self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8'), 1)
        self.img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.img_bgr_undist = cv.undistort(self.img_bgr, self.A, self.dist_coeffs)
        self.img_bgr_undist = self.img_bgr
        self.img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        self.img_rgb_undist = cv.cvtColor(self.img_bgr_undist, cv.COLOR_BGR2RGB)
        
        self.process_or_not()
        
    
    def depth_callback(self, msg):
        '''
        Callback function for the depth image subscriber.
        Converts the ROS image message to OpenCV format and undistorts it.
        '''
        self.img_depth = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # self.img_depth = cv.flip(self.img_depth, 1)
        
        self.process_or_not()
    
    def listener_callback(self):
        '''
        Callback function for processing the images.
        Segments the image using Mahalanobis distance and finds contours.
        Annotates the largest contour and publishes the target position and reference frame.
        Can print out the pixel coordinates, camera frame coordinates, contour radius, and area.
        Can also visualize the RGB image with contour annotations, the segmented image, and the depth image.
        '''
        
        kernel_size = (5, 5)
        self.img_rgb_blurred = cv.GaussianBlur(self.img_rgb, (kernel_size[0], kernel_size[1]), 0)
        
        tolerance_mahalanobis = 10

        # Segment the image using Mahalanobis distance
        segmented_image_mahalanobis = imgproc.segment_image_by_color_distance(
            self.img_rgb_blurred, self.pingpong_color_data_rgb[2], None, tolerance_mahalanobis, "Mahalanobis"
        )

        # Morphological filtering
        kernel = np.ones((kernel_size[0], kernel_size[1]), np.uint8)
        segmented_image_mahalanobis_filtered = cv.morphologyEx(segmented_image_mahalanobis, cv.MORPH_OPEN, kernel)
        segmented_image_mahalanobis_filtered = cv.morphologyEx(segmented_image_mahalanobis_filtered, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(segmented_image_mahalanobis_filtered, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        # Annotate largest contour
        min_contour_area = 0 # 100
        if contours:
            # Assume the largest contour is the object of interest
            contour = max(contours, key=cv.contourArea)
            
            if cv.contourArea(contour) > min_contour_area:
                # Get the centroid of the contour (x, y)
                M = cv.moments(contour)
                if M['m00'] != 0:  # Avoid division by zero
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    
                    (x, y), radius = cv.minEnclosingCircle(contour)
                    radius = int(radius)
                    area = cv.contourArea(contour)
                    
                    # Add the new (x, y) to the sliding window
                    self.xy_window.append((cX, cY))

                    # If the window is filled, apply the sliding window average and prediction
                    if len(self.xy_window) == self.window_size:
                        # If the predicted (x, y) is outside the tolerance, use the predicted value
                        if self.xy_predicted is not None:
                            pred_x, pred_y = self.xy_predicted
                            if abs(pred_x - cX) > self.tolerance or abs(pred_y - cY) > self.tolerance:
                                cX, cY = pred_x, pred_y  # Use predicted coordinates

                        # Predict next (x, y) based on simple linear regression (or a more sophisticated model)
                        if len(self.xy_window) == self.window_size:
                            pred_x, pred_y = self.predict_next_xy()
                            self.xy_predicted = (pred_x, pred_y)
                            
                        # Ensure pixel is within bounds
                        depth_z_m = 0.0

                        # Create a mask from the contour
                        mask = np.zeros(self.img_depth.shape, dtype=np.uint8)
                        cv.drawContours(mask, [contour], -1, color=255, thickness=-1)

                        # Get valid depth values within the contour mask
                        depth_vals = self.img_depth[mask == 255]
                        depth_vals = depth_vals[depth_vals > 0]  # filter out invalid depth
                        depth_vals = depth_vals[np.isfinite(depth_vals)]
                        real_area = 0.0
                        if len(depth_vals) > 0:
                            # Use median for robustness
                            depth_z = np.median(depth_vals)
                            depth_z_m = depth_z / 1000.0  # convert mm to meters

                            # Calculate pixel-to-meters conversion from intrinsics
                            pixel_area_m2 = (depth_z_m / self.A[0, 0]) * (depth_z_m / self.A[1, 1])
                            real_area = pixel_area_m2 * area * 10 # convert pixel area to m^2
                        if real_area > 0: # 0.001
                            # Get the camera frame coordinates (X, Y, Z)
                            R = np.eye(3)  # Assuming no rotation
                            t = np.zeros((3, 1))  # Assuming no translation
                            T = np.hstack((R, t))  # Combine rotation and translation into a 3x4 matrix
                            M = self.pixel_to_camera((cX, cY), self.A, depth_z_m, T)
                            
                            # Publish the target position
                            target_position = Point()
                            target_position.x = M[0]    
                            target_position.y = M[1]
                            target_position.z = M[2]
                            self.pos_pub.publish(target_position)
                            
                            # Publish the target reference frame
                            target_reference_frame = Float64MultiArray()
                            target_reference_frame.data = T.flatten().tolist()
                            self.transform_pub.publish(target_reference_frame)
                            
                            self.get_logger().info(f"Pixel Coordinates:")
                            self.get_logger().info(f"({cX}, {cY})")
                            self.get_logger().info(f"X,Y,Z Coordinates:")
                            self.get_logger().info(f"({M[0]:.3f}, {M[1]:.3f}, {M[2]:.3f})")
                            self.get_logger().info(f"Contour Radius: {radius}")
                            self.get_logger().info(f"Contour Area: {area}")
                            self.get_logger().info(f"Real Area: {real_area:.10f} m^2")
                            self.get_logger().info(f"")
                            
                            cv.circle(self.img_rgb_undist, (cX, cY), 10, (0, 255, 0), -1)  # Draw the center
                            (x, y), radius = cv.minEnclosingCircle(contour)
                            cv.circle(self.img_rgb_undist, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                            # Write pixel coordinates (x,y) and camera frame coordinates (X,Y,Z)
                            cv.putText(self.img_rgb_undist, f"({cX}, {cY})",
                                    (cX + 5, cY - 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv.putText(self.img_rgb_undist, f"({M[0]:.3f}, {M[1]:.3f}, {M[2]:.3f})",
                                    (cX + 5, cY + 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        depth_clipped = np.where((self.img_depth > 0) & (self.img_depth < 1500), self.img_depth, 0)
        depth_normalized = cv.normalize(depth_clipped, None, 0, 255, cv.NORM_MINMAX)
        depth_gray = depth_normalized.astype(np.uint8)
        depth_colormap = cv.applyColorMap(depth_gray, cv.COLORMAP_JET)
        
        # Show images
        cv.imshow("Undistorted RGB Image", cv.cvtColor(self.img_rgb_undist, cv.COLOR_RGB2BGR))
        cv.moveWindow("Undistorted RGB Image", 0, 0)  # Move to top-left corner of the screen
        cv.imshow("Segmented Image", segmented_image_mahalanobis_filtered)
        cv.moveWindow("Segmented Image", 640, 0)  # Move to the right of the first window (adjust width as needed)
        cv.imshow("Depth Image", depth_colormap)
        cv.moveWindow("Depth Image", 1280, 0)

        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    node.get_logger().info('Image Subscriber Node has been started.')

    # Spin the node to keep it active
    rclpy.spin(node)

    # Shutdown and cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()