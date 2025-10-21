metainfo = dict(
    dataset_name="spinetrack",
    paper_info=dict(
        author="Khan, Muhammad Saif Ullah and Krauß, Stephan and Stricker, Didier",
        title="Towards Unconstrained 2D Pose Estimation of the Human Spine",
        container="CVPRW",
        year="2025",
        homepage="https://github.com/dfki-av/spinepose",
    ),
    keypoint_info={
        0: dict(name="nose", id=0, color=[204, 51, 51], type="upper", swap=""),
        1: dict(name="left_eye", id=1, color=[204, 51, 51], type="upper", swap="right_eye"),
        2: dict(name="right_eye", id=2, color=[204, 51, 51], type="upper", swap="left_eye"),
        3: dict(name="left_ear", id=3, color=[204, 51, 51], type="upper", swap="right_ear"),
        4: dict(name="right_ear", id=4, color=[204, 51, 51], type="upper", swap="left_ear"),
        5: dict(name="left_shoulder", id=5, color=[204, 51, 51], type="upper", swap="right_shoulder"),
        6: dict(name="right_shoulder", id=6, color=[204, 51, 51], type="upper", swap="left_shoulder"),
        7: dict(name="left_elbow", id=7, color=[204, 51, 51], type="upper", swap="right_elbow"),
        8: dict(name="right_elbow", id=8, color=[204, 51, 51], type="upper", swap="left_elbow"),
        9: dict(name="left_wrist", id=9, color=[204, 51, 51], type="upper", swap="right_wrist"),
        10: dict(name="right_wrist", id=10, color=[204, 51, 51], type="upper", swap="left_wrist"),
        11: dict(name="left_hip", id=11, color=[204, 51, 51], type="lower", swap="right_hip"),
        12: dict(name="right_hip", id=12, color=[204, 51, 51], type="lower", swap="left_hip"),
        13: dict(name="left_knee", id=13, color=[204, 51, 51], type="lower", swap="right_knee"),
        14: dict(name="right_knee", id=14, color=[204, 51, 51], type="lower", swap="left_knee"),
        15: dict(name="left_ankle", id=15, color=[204, 51, 51], type="lower", swap="right_ankle"),
        16: dict(name="right_ankle", id=16, color=[204, 51, 51], type="lower", swap="left_ankle"),
        17: dict(name="head", id=17, color=[204, 51, 51], type="upper", swap=""),
        18: dict(name="neck", id=18, color=[36, 93, 48], type="upper", swap=""),
        19: dict(name="hip", id=19, color=[35, 161, 210], type="lower", swap=""),
        20: dict(name="left_big_toe", id=20, color=[204, 51, 51], type="lower", swap="right_big_toe"),
        21: dict(name="right_big_toe", id=21, color=[204, 51, 51], type="lower", swap="left_big_toe"),
        22: dict(name="left_small_toe", id=22, color=[204, 51, 51], type="lower", swap="right_small_toe"),
        23: dict(name="right_small_toe", id=23, color=[204, 51, 51], type="lower", swap="left_small_toe"),
        24: dict(name="left_heel", id=24, color=[204, 51, 51], type="lower", swap="right_heel"),
        25: dict(name="right_heel", id=25, color=[204, 51, 51], type="lower", swap="left_heel"),
        26: dict(name="spine_01", id=26, color=[248, 144, 171], type="upper", swap=""),
        27: dict(name="spine_02", id=27, color=[248, 144, 171], type="upper", swap=""),
        28: dict(name="spine_03", id=28, color=[248, 144, 171], type="upper", swap=""),
        29: dict(name="spine_04", id=29, color=[194, 134, 91], type="upper", swap=""),
        30: dict(name="spine_05", id=30, color=[194, 134, 91], type="upper", swap=""),
        31: dict(name="left_latissimus", id=31, color=[204, 51, 51], type="upper", swap="right_latissimus"),
        32: dict(name="right_latissimus", id=32, color=[204, 51, 51], type="upper", swap="left_latissimus"),
        33: dict(name="left_clavicle", id=33, color=[204, 51, 51], type="upper", swap="right_clavicle"),
        34: dict(name="right_clavicle", id=34, color=[204, 51, 51], type="upper", swap="left_clavicle"),
        35: dict(name="neck_02", id=35, color=[36, 93, 48], type="upper", swap=""),
        36: dict(name="neck_03", id=36, color=[36, 93, 48], type="upper", swap=""),
    },
    skeleton_info={
        0: dict(link=("left_ankle", "left_knee"), id=0, color=[0, 255, 0]),
        1: dict(link=("left_knee", "left_hip"), id=1, color=[0, 255, 0]),
        2: dict(link=("left_hip", "hip"), id=2, color=[0, 255, 0]),
        3: dict(link=("right_ankle", "right_knee"), id=3, color=[255, 128, 0]),
        4: dict(link=("right_knee", "right_hip"), id=4, color=[255, 128, 0]),
        5: dict(link=("right_hip", "hip"), id=5, color=[255, 128, 0]),
        6: dict(link=("head", "neck_03"), id=6, color=[51, 153, 255]),
        7: dict(link=("neck_03", "neck_02"), id=7, color=[51, 153, 255]),
        8: dict(link=("neck_02", "neck"), id=8, color=[51, 153, 255]),
        9: dict(link=("neck", "spine_05"), id=9, color=[51, 153, 255]),
        10: dict(link=("spine_05", "spine_04"), id=10, color=[51, 153, 255]),
        11: dict(link=("spine_04", "spine_03"), id=11, color=[51, 153, 255]),
        12: dict(link=("spine_03", "spine_02"), id=12, color=[51, 153, 255]),
        13: dict(link=("spine_02", "spine_01"), id=13, color=[51, 153, 255]),
        14: dict(link=("spine_01", "hip"), id=14, color=[51, 153, 255]),
        18: dict(link=("left_shoulder", "left_elbow"), id=18, color=[0, 255, 0]),
        19: dict(link=("left_elbow", "left_wrist"), id=19, color=[0, 255, 0]),
        20: dict(link=("left_shoulder", "right_shoulder"), id=20, color=[255, 128, 0]),
        21: dict(link=("right_shoulder", "right_elbow"), id=21, color=[255, 128, 0]),
        22: dict(link=("right_elbow", "right_wrist"), id=22, color=[255, 128, 0]),
        23: dict(link=("left_eye", "right_eye"), id=23, color=[51, 153, 255]),
        24: dict(link=("head", "nose"), id=24, color=[51, 153, 255]),
        25: dict(link=("nose", "left_eye"), id=25, color=[51, 153, 255]),
        26: dict(link=("nose", "right_eye"), id=26, color=[51, 153, 255]),
        27: dict(link=("left_eye", "left_ear"), id=27, color=[51, 153, 255]),
        28: dict(link=("right_eye", "right_ear"), id=28, color=[51, 153, 255]),
        29: dict(link=("left_ear", "left_shoulder"), id=29, color=[51, 153, 255]),
        30: dict(link=("right_ear", "right_shoulder"), id=30, color=[51, 153, 255]),
        31: dict(link=("left_ankle", "left_big_toe"), id=31, color=[0, 255, 0]),
        32: dict(link=("left_ankle", "left_small_toe"), id=32, color=[0, 255, 0]),
        33: dict(link=("left_ankle", "left_heel"), id=33, color=[0, 255, 0]),
        34: dict(link=("right_ankle", "right_big_toe"), id=34, color=[255, 128, 0]),
        35: dict(link=("right_ankle", "right_small_toe"), id=35, color=[255, 128, 0]),
        36: dict(link=("right_ankle", "right_heel"), id=36, color=[255, 128, 0]),
    },
    # the joint_weights is modified by MMPose Team
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
    ]
    + [1.0, 1.5, 1.5]  # Head Top, Neck Base, Hip
    + [1.2] * 6  # Feet
    + [1.2, 1.2, 1.2, 1.2, 1.2]  # Spine_01 to Spine_05
    + [1.0, 1.0]  # Latissimus Dorsi
    + [1.2, 1.2]  # Clavicles
    + [1.5, 1.5],  # Neck_02, Neck_03
    # 'https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/'
    # 'HalpeCOCOAPI/PythonAPI/halpecocotools/cocoeval.py#L245'
    sigmas=[
        # COCO
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        # Central Axix (Halpe/COCO-Wholebody)
        0.026,  # head = head top
        0.04,  # neck = neck base / C7, increased from 0.026 to 0.04
        0.07,  # hip = sacrum midpoint, increased from 0.066 to 0.07
        # Feet
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
        # Spine
        0.10,  # spine_01 = L5, As with L3, slight ambiguity remains.
        0.10,  # spine_02 = L3, Lower back landmarks tend to be less defined, so a similar value is used.
        0.10,  # spine_03 = T12/L1, The transition zone is less distinct, warranting a higher uncertainty.
        0.08,  # spine_04 = T8, Similar reasoning as T3.
        0.08,  # spine_05 = T3, Increased ambiguity from curvature and soft tissue leads to a moderate sigma.
        # Latissimus Dorsi
        # Rationale: Given the diffuse boundaries of these muscles, a higher sigma is appropriate
        # to capture the inherent annotation variability.
        0.12,
        0.12,
        # Clavicles
        # Rationale: These bony landmarks are usually well defined but can vary with shoulder
        # pose; 0.07 reflects moderate confidence slightly lower than the shoulder joints (0.079).
        0.07,
        0.07,
        # Neck
        0.07,  # neck_02 = C3/C4
        0.06,  # neck_03 = C1
    ],
)
