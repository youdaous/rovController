#!/usr/bin/env python
#coding=utf-8

from sys import path
path.append(r"/home/youda/casadi-linux-py27-fadc864")
import math
import rospy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import WrenchStamped
from uuv_control_interfaces.vehicle import Vehicle

def thrust():
    rospy.init_node('thrust_test')
    thrust_pub = rospy.Publisher('/bluerov2/thruster_manager/input_stamped', WrenchStamped, queue_size=1)
    rate = rospy.Rate(10)

    force_msg = WrenchStamped()
    force_msg.header.stamp = rospy.Time.now()
    force_msg.wrench.force.x = 0
    force_msg.wrench.torque.z = 0

    while not rospy.is_shutdown():
        thrust_pub.publish(force_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        thrust()
    except rospy.ROSInterruptException:
        pass