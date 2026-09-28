#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage.graph import route_through_array


"""
@brief Thresholds a black and white image by brightness where darker pixels become white

@param frame The thresholded image.
@param threshold the brighness to threshold against.
@return frame. where frame is the thresholded image
"""
def threshold(frame, threshold):
  ret,thresh = cv2.threshold(frame,threshold,255,cv2.THRESH_BINARY_INV)
  return thresh

"""
@brief Finds a centered path and point dist pixels along the path
from a thresholded black and white image

@param frame The thresholded image.
@param dist number of pixels along path to return point of.
@return (x, y), frame. where (x, y) are the coordinates of the point and frame is an image with just the centered path
(x, y) will == (-1, -1) if no path found
"""
def get_line_and_point(frame, dist):
    mask = (frame > 0).astype(np.uint8)

    # find all connected componenets of the mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # No white components
    if n <= 1:
        return (-1, -1), np.zeros_like(frame)

    #find the largest connected segment
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = labels == largest

    #Pad the image to make the edges count as background
    #find the distance of each pixel to the background
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]

    # Cost function to optimize for the furthest distance from the edges
    cost = 1 + 10 / (d + 1)
    cost[~mask] = np.inf

    ys, xs = np.where(mask)

    valid_rows = np.where(mask.any(axis=1))[0]
    sy = valid_rows[-1]
    row_xs = np.where(mask[sy])[0]
    # start at middle pixel in bottom row of image
    sx = row_xs[len(row_xs) // 2]

    start = (sy, sx)

    #Find the pixel that is furthest from the start as a goal
    i = np.argmax((ys - sy)**2 + (xs - sx)**2)
    goal = (ys[i], xs[i])

    path, _ = route_through_array(
        cost,
        start,
        goal,
        fully_connected=True
    )
    path = np.array(path)

    circle = path[min(dist, len(path) - 1)]

    out = np.zeros_like(frame)
    out[path[:, 0], path[:, 1]] = 255

    return circle, out

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw',
                                                  self.callback, 1)
        self.image_saved = False
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.last_circle = (0, 0)

        self.kP = 0.01
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        thresh_frame = threshold(grey, 100)

        circle, thin_line = get_line_and_point(thresh_frame, 40)
        circle = tuple(circle)
        # continue from last circle position if none found
        if circle == (-1, -1):
            circle = self.last_circle
        else:
            circle = tuple(circle)
            self.last_circle = circle
        self.get_logger().info(str(circle))
        self.move = Twist()
        self.move.linear.x = 0.5
        # circle = (y, x)
        self.move.angular.z = self.kP * (cv_image.shape[0] // 2 - circle[1])
        self.cmd_vel_pub.publish(msg=self.move)

def main(args=None):
    rclpy.init(args=args)
    lf = LineFollower()
    lf.get_logger().info("Starting...")
    try:
        rclpy.spin(lf)
    except KeyboardInterrupt:
        lf.get_logger().info("Shutting down")
    finally:
        cv2.destroyAllWindows()
        lf.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
