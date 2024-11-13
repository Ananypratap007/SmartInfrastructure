#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32  # Adjust according to MuSHR’s message types
import torch  # or another library you used for RL

class RLNavigator:
    def __init__(self, policy_path):
        self.policy = torch.load(policy_path)  # Load trained policy
        self.state_sub = rospy.Subscriber('/state_topic', Float32, self.state_callback)
        self.action_pub = rospy.Publisher('/cmd_vel', Float32, queue_size=10)

    def state_callback(self, state_msg):
        state = self.process_state(state_msg)
        action = self.policy(state)
        self.action_pub.publish(action)

    def process_state(self, state_msg):
        # Process the state message to fit the RL model's input format
        return torch.tensor([state_msg.data])

if __name__ == '__main__':
    rospy.init_node('rl_navigator')
    navigator = RLNavigator('/path/to/policy.pth')
    rospy.spin()
