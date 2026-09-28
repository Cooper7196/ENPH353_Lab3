#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Circle_Driver_Node(Node):
   def __init__(self):
       super().__init__(node_name='driver_node')
       self.publisher_ = self.create_publisher(msg_type=Twist, topic='/cmd_vel', qos_profile=1)
       time.sleep(1)
       self.move = Twist()
       self.move.linear.x = 0.5
       self.move.angular.z = 0.3
       self.publisher_.publish(msg=self.move)
       # END

def main(args=None):
    rclpy.init(args=args)
    cdn = Circle_Driver_Node()
    rclpy.spin(cdn)
    rclpy.shutdown()

if __name__== '__main__':
    main()