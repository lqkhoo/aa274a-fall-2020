#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose2D, PoseStamped
from visualization_msgs.msg import Marker

class VisGoal(object):

    def __init__(self):
        rospy.init_node('section5_visgoal', anonymous=True)

        self.QUEUE_SIZE = 10

        # Messaging
        self.pub = rospy.Publisher('marker_topic', Marker, queue_size=self.QUEUE_SIZE)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.sub_callback)

        # State
        self.has_data = False

        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.header.stamp = rospy.Time()
        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        self.marker.id = 0
        self.marker.type = 2 # sphere
        self.marker.pose.position.x = 1
        self.marker.pose.position.y = 1
        self.marker.pose.position.z = 0.2
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0


    ## Callbacks

    def sub_callback(self, posestamped):
        # rospy.loginfo(posestamped)
        self.marker.pose.position.x = posestamped.pose.position.x
        self.marker.pose.position.y = posestamped.pose.position.y
        self.has_data = True

    def shutdown_callback(self):
        pass # do nothing

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():

            if self.has_data:
                self.pub.publish(self.marker)
        
            rate.sleep()



if __name__ == '__main__':    
    node = VisGoal()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()