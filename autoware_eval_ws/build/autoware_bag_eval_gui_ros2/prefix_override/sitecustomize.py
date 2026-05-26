import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gazi/Desktop/autoware_eval_ws/install/autoware_bag_eval_gui_ros2'
