from enum import Enum

from calipod.trackers.charuco_tracker import CharucoTracker
from calipod.trackers.hand_tracker import HandTracker
from calipod.trackers.holistic.holistic_tracker import HolisticTracker
from calipod.trackers.pose_tracker import PoseTracker
from calipod.trackers.simple_holistic_tracker import SimpleHolisticTracker
from calipod.trackers.fly_tracker import FlyTracker

# Temporarily removed face tracker because crashing sample project
# from calipod.trackers.face_tracker import FaceTracker


class TrackerEnum(Enum):
    HAND = HandTracker
    FLY = FlyTracker
    POSE = PoseTracker
    SIMPLE_HOLISTIC = SimpleHolisticTracker
    HOLISTIC = HolisticTracker
    CHARUCO = CharucoTracker
    # FACE = FaceTracker


if __name__ == "__main__":
    tracker_factories = [enum_member.name for enum_member in TrackerEnum]
    print(tracker_factories)

