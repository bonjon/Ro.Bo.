#!/usr/bin/env python3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from enum import Enum
from panda3d.core import Material, DirectionalLight, Vec3, Vec4, Mat4, Mat3, LPoint3, ClientBase, NodePath, LineSegs, GeomVertexWriter, PointLight, AmbientLight, Quat

from exts.ik.CCDIK.ik_actor import IKActor, IKChain

import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class CustomLandmark(Enum):
    ShoulderCenter = 100,
    HipCenter = 101,
    Offset = 102,


def center_point(a: Vec3, b: Vec3):
    return (a + b) / 2


def transform_to_hip_origin(world_landmarks):
    hip: Vec3 = center_point(world_landmarks[23][1], world_landmarks[24][1])
    world_landmarks = [[idx, lm - hip] for idx, lm in world_landmarks]
    world_landmarks.append([CustomLandmark.Offset, hip])
    return world_landmarks


def saacos(f: float) -> float:
    if f <= -1.0:
        return np.pi
    if f >= 1.0:
        return 0

    return np.arccos(f)


def axis_angle_normalized_to_quat(axis: Vec3, angle: float) -> Quat:
    phi = 0.5 * angle
    si = np.sin(phi)
    co = np.cos(phi)
    q = Quat(co, axis * si)
    return q


def vec_to_track_quat(vec: Vec3, axis: Vec3, up: Vec3):
    vec: Vec3 = -vec

    eps = 1e-4
    normal: Vec3 = Vec3()
    tvec: Vec3 = Vec3()

    angle = 0
    si = 0
    co = 0
    len = vec.length()

    if axis.x == -1:
        tvec = vec
        axis.x = 1
    elif axis.y == -1:
        tvec = vec
        axis.y = 1
    elif axis.z == -1:
        tvec = vec
        axis.z = 1
    else:
        tvec = -vec

    if axis.x == 1:
        normal = Vec3(0, -tvec.y, tvec.x)
        if (abs(tvec.y) + abs(tvec.z)) < eps:
            normal.y = 1
        co = tvec.x
    elif axis.y == 1:
        normal = Vec3(tvec.z, 0, -tvec.x)
        if (abs(tvec.x) + abs(tvec.z)) < eps:
            normal.z = 1
        co = tvec.y
    else:
        normal = Vec3(-tvec.y, tvec.x, 0)
        if (abs(tvec.x) + abs(tvec.y)) < eps:
            normal.x = 1
        co = tvec.z

    co /= len

    normal.normalize()

    quat = axis_angle_normalized_to_quat(normal, saacos(co))

    if axis != up:
        mat: Mat3 = Mat3()
        quat2: Quat = Quat()
        quat.extractToMatrix(mat)

        if axis.x == 1:
            if up.y == 1:
                angle = 0.5 * np.arctan2(mat[2][2], mat[2][1])
            else:
                angle = -0.5 * np.arctan2(mat[2][1], mat[2][2])
        elif axis.y == 1:
            if up.x == 1:
                angle = -0.5 * np.arctan2(mat[2][2], mat[2][0])
            else:
                angle = 0.5 * np.arctan2(mat[2][0], mat[2][2])
        else:
            if up.x == 1:
                angle = 0.5 * np.arctan2(-mat[2][1], -mat[2][0])
            else:
                angle = -0.5 * np.arctan2(-mat[2][0], -mat[2][1])

        co = np.cos(angle)
        si = np.sin(angle) / len
        quat2.set(co, tvec.x * si, tvec.y * si, tvec.z * si)

        quat = quat * quat2

    return quat


def rotate_towards(source: Vec3, dest: Vec3, up: Vec3 = Vec3(0, 1, 0), axis: Vec3 = Vec3(0, 0, 1)) -> Quat:
    direction: Vec3 = dest - source
    direction.normalize()
    # direction = -direction
    # s = axis
    # s.normalize()
    # u = s.cross(direction)
    #
    # rotation_mat: Mat4 = Mat4()
    # rotation_mat.setCol(0, s)
    # rotation_mat.setCol(1, u)
    # rotation_mat.setCol(2, -direction)
    # rotation_mat.setCol(3, Vec4(0, 0, 0, 1))
    # translation_mat: Mat4 = Mat4()
    # translation_mat.setCol(3, source)
    # look_at = rotation_mat  # * translation_mat
    #
    # quat = Quat()
    # quat.setFromMatrix(look_at)

    # return quat
    return vec_to_track_quat(direction, axis, up)


def normal_from_plane(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    return (b - a).cross(c - a)


def limb_rotations(world_landmarks):
    res = []
    lms = mp_pose.PoseLandmark

    def calc_chain_rotations(data):
        rotations = []
        for i in range(1, len(data)):
            # print(data[i][1], data[i - 1][1])
            # quart = rotate_towards(
            #     data[i][1], data[i - 1][1], '-Y', 'Z')
            # source = Vec3(data[i][1].x, data[i][1].y, data[i][1].z)
            # dest = Vec3(data[i - 1][1].x, data[i - 1][1].y, data[i - 1][1].z)
            quart = rotate_towards(
                data[i][1], data[i - 1][1], axis=Vec3(0, -1, 0), up=Vec3(0, 0, 1))
            rotations.append([data[i - 1][0], quart])
        return rotations

    left_leg = [world_landmarks[23], world_landmarks[25],
                world_landmarks[27]]
    right_leg = [world_landmarks[24], world_landmarks[26], world_landmarks[28]]
    left_arm = [world_landmarks[lms.LEFT_SHOULDER],
                world_landmarks[lms.LEFT_ELBOW], world_landmarks[lms.LEFT_WRIST]]
    right_arm = [world_landmarks[lms.RIGHT_SHOULDER],
                 world_landmarks[lms.RIGHT_ELBOW], world_landmarks[lms.RIGHT_WRIST]]
    for limbs in [left_arm, right_arm, left_leg, right_leg]:
        res += calc_chain_rotations(limbs)
    return res


def torso_rotations(world_landmarks):
    res = []

    left_hip = world_landmarks[23][1]
    right_hip = world_landmarks[24][1]
    hip_center = center_point(left_hip, right_hip)

    shoulder_center = center_point(
        world_landmarks[11][1], world_landmarks[12][1])

    normal: Vec3 = normal_from_plane(
        world_landmarks[23][1], world_landmarks[24][1], shoulder_center)
    normal.normalize()

    tangent = hip_center - right_hip
    tangent.normalize()

    binormal = hip_center - shoulder_center
    binormal.normalize()

    matrix = Mat4()
    matrix.setRow(0, tangent)
    matrix.setRow(1, normal)
    matrix.setRow(2, binormal)
    matrix.setRow(3, Vec4(0, 0, 0, 1))

    quat = Quat()
    quat.setFromMatrix(matrix)

    res.append([CustomLandmark.HipCenter, quat])

    return res


def calculate_inverse_kinematics(world_landmarks):
    joint_rotations = []
    joint_scaling = []

    shoulder_center = center_point(
        world_landmarks[11][1], world_landmarks[12][1])
    hip_center = center_point(
        world_landmarks[23][1], world_landmarks[24][1])
    world_landmarks.append([CustomLandmark.HipCenter, hip_center])
    world_landmarks.append([CustomLandmark.ShoulderCenter, shoulder_center])
    world_landmarks = transform_to_hip_origin(world_landmarks)

    joint_rotations += limb_rotations(world_landmarks)
    joint_rotations += torso_rotations(world_landmarks)

    # print(joint_rotations)
    return world_landmarks, joint_rotations, joint_scaling


# import mediapipe

# def convert_coordinate_system(p_from: LPoint3, frame):

LANDMARK_SCALE = 10

IK_CHAIN_INFOS = {
    mp_pose.PoseLandmark.LEFT_WRIST: {
        # 'joint_8', 'joint_7', 'joint_4'],
        'joints': ['joint_9', 'joint_5', 'joint_5_dup_1', 'joint_8', 'joint_7', 'joint_4'],
        'static': ['joint_5'],
        'target_name': 'right_wrist',
        'constraints': [
            # {
            #     'type': 1,
            #     'joint': 'joint_5_dup_0',
            #     'unit': Vec3(0, 0, 1),
            #     'min_angle': 0,
            #     'max_angle': np.pi * 0.05
            # },
            # {
            #     'type': 1,
            #     'joint': 'joint_9',
            #     'unit': Vec3(0, 0, 1),
            #     'min_angle': 0,
            #     'max_angle': np.pi * 0.05
            # },
            # {
            #     'type': 1,
            #     'joint': 'joint_0',
            #     'unit': Vec3(1, 0, 0),
            #     'min_angle': -np.pi * 0.1,
            #     'max_angle': np.pi * 0.1
            # },
            # {
            #     'type': 1,
            #     'joint': 'joint_9',
            #     'unit': Vec3(1, 0, 0),
            #     'min_angle': -np.pi * 0.1,
            #     'max_angle': np.pi * 0.01
            # },
            # {
            #     'type': 1,
            #     'joint': 'joint_14',
            #     'unit': Vec3(0, 1, 0),
            #     'min_angle': -np.pi * 0.75,
            #     'max_angle': np.pi * 0.75
            # },
            # {
            #     'type': 1,
            #     'joint': 'joint_12',
            #     'unit': Vec3(0, 0, 1),
            #     'min_angle': -np.pi * 0.5,
            #     'max_angle': np.pi * 0.5
            # },
        ]
    },

    # mp_pose.PoseLandmark.LEFT_WRIST: {
    #     'joints': ['joint_3', 'joint_2', 'joint_0'],
    #     # 'static': 'Shoulder.L',
    #     'target_name': 'left_wrist',
    #     'constraints': [
    #         # {
    #         #     'type': 1,
    #         #     'joint': 'joint_3',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': np.pi * 0.05,
    #         #     'max_angle': np.pi * 0.05
    #         # },
    #         # {
    #         #     'type': 0,
    #         #     'joint': 'joint_2',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #         # {
    #         #     'type': 0,
    #         #     'joint': 'joint_0',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #     ]
    # },
    # mp_pose.PoseLandmark.RIGHT_ANKLE: {
    #     'joints': ['joint_16', 'joint_18', 'joint_17'],
    #     # 'static': 'Shoulder.L',
    #     'target_name': 'right_ankle',
    #     'constraints': [
    #         # {
    #         #     'type': 1,
    #         #     'joint': 'joint_3',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': np.pi * 0.05,
    #         #     'max_angle': np.pi * 0.05
    #         # },
    #         # {'joint_1'
    #         #     'type': 0,
    #         #     'joint': 'joint_2',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #         # {
    #         #     'type': 0,
    #         #     'joint': 'joint_0',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #     ]
    # },
    #
    # mp_pose.PoseLandmark.LEFT_ANKLE: {
    #     'joints': ['joint_4', 'joint_6', 'joint_5'],
    #     # 'static': 'Shoulder.L',
    #     'target_name': 'left_ankle',
    #     'constraints': [
    #         # {
    #         #     'type': 1,
    #         #     'joint': 'joint_3',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': np.pi * 0.05,
    #         #     'max_angle': np.pi * 0.05
    #         # },
    #         # {'joint_1'
    #         #     'type': 0,
    #         #     'joint': 'joint_2',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #         # {
    #         #     'type': 0,
    #         #     'joint': 'joint_0',
    #         #     'unit': Vec3(0, 1, 0),
    #         #     'min_angle': -np.pi * 0.5,
    #         #     'max_angle': np.pi * 0.5
    #         # },
    #     ]
    # }
}

joint_mapping = {
    mp_pose.PoseLandmark.NOSE: 'joint_11',
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 'joint_15',
    mp_pose.PoseLandmark.RIGHT_ELBOW: 'joint_14',
    mp_pose.PoseLandmark.RIGHT_WRIST: 'joint_12',

    mp_pose.PoseLandmark.LEFT_SHOULDER: 'joint_3',
    mp_pose.PoseLandmark.LEFT_ELBOW: 'joint_2',
    mp_pose.PoseLandmark.LEFT_WRIST: 'joint_0',

    mp_pose.PoseLandmark.RIGHT_HIP: 'root_dup_0',
    mp_pose.PoseLandmark.RIGHT_KNEE: 'joint_6',
    mp_pose.PoseLandmark.RIGHT_ANKLE: 'joint_5',

    mp_pose.PoseLandmark.LEFT_HIP: 'root_dup_2',
    mp_pose.PoseLandmark.LEFT_KNEE: 'joint_18',
    mp_pose.PoseLandmark.LEFT_ANKLE: 'joint_17',

    # CustomLandmark.ShoulderCenter: '''
    CustomLandmark.HipCenter: 'root_dup_1',
}


class Animate(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.dlight = PointLight('my dlight')
        self.dlnp = self.render.attachNewNode(self.dlight)
        self.dlnp.setPos(0, -10, 0)
        self.render.setLight(self.dlnp)
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.rootNode = self.render.attachNewNode("Torso")
        self.mp_nodes_root = self.render.attachNewNode("MPRoot")
        self.mp_nodes_root.setHpr(0, -90, 0)
        self.mp_nodes_root.setPos(0, 0, 5)
        self.mp_nodes_root.reparentTo(self.render)
        # self.rootNode.setScale(8, 8, 8)
        self.rootNode.setPos(0, 0, -5)
        # self.rootNode.setHpr(0, 90, 0)
        self.render.setLight(alnp)
        self.model = Actor('giua.glb')
        self.ikmodel = IKActor(self.model)
        # self.ikmodel.actor.setHpr(0, 0, 0)
        # self.ikmodel.actor.setScale(10, 10, 10)
        # self.ikmodel.actor.setPos(0, 0, 0)
        self.ikmodel.reparent_to(self.rootNode)
        print(self.ikmodel.actor)

        self.camera.setPos(0, -10, -10)
        self.material = Material()
        self.material.setShininess(5.0)  # Make this material shiny
        self.material.setAmbient((0, 0, 1, 1))  # Make this material blue
        self.model.setMaterial(self.material)
        self.axis = self.loader.loadModel('axis.glb')
        # self.axis.reparentTo(self.render)
        self.video = cv2.VideoCapture(0)

        self.mp_nodes = {}
        self.initMPLandmarks()

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)

        self.counter = 0

        # self.taskMgr.add(self.spinJoint, "SpinJoint")
        self.taskMgr.add(self.poseEstimation, "Pose Estimation")

    def initMPLandmarks(self):
        self.joints = {}

        # for joint in self.ikmodel.actor.getJoints():
        # self.joints[joint.getName()] = self.model.controlJoint(
        #     None, "modelRoot", joint.getName())

        lms = mp_pose.PoseLandmark

        def setLandmark(lm):
            material = Material()
            material.setShininess(5.0)  # Make this material shiny
            material.setAmbient((0, 0, 1, 1))  # Make this material blue
            self.mp_nodes[lm] = self.loader.loadModel('sphere.glb')
            self.mp_nodes[lm].setMaterial(material)

        def reparentTo(node, parent):
            node.setScale(0.25)
            node.reparentTo(self.mp_nodes_root)
            # node.reparentTo(parent)

        def initIKChain(lm, ikInfo):
            chain = self.ikmodel.create_ik_chain(ikInfo['joints'])
            target: NodePath = self.render.attachNewNode(
                ikInfo['target_name'] + "_target")
            chain.set_target(self.mp_nodes[lm])
            if 'static' in ikInfo:
                for joint in ikInfo['static']:
                    chain.set_static(joint)
            for constraint in ikInfo['constraints']:
                if constraint['type'] == 0:
                    # Ball constraint
                    chain.set_ball_constraint(
                        constraint['joint'], min_ang=constraint['min_angle'], max_ang=constraint['max_angle'])
                else:
                    chain.set_hinge_constraint(
                        constraint['joint'],
                        constraint['unit'],
                        min_ang=constraint['min_angle'],
                        max_ang=constraint['max_angle'])
            chain.debug_display()
            return (chain, target)

        setLandmark(lms.NOSE)
        self.mp_nodes[lms.NOSE].setScale(0.25)
        self.mp_nodes[lms.NOSE].reparentTo(self.mp_nodes_root)

        setLandmark(lms.RIGHT_SHOULDER)
        reparentTo(self.mp_nodes[lms.RIGHT_SHOULDER], self.mp_nodes[lms.NOSE])

        setLandmark(lms.RIGHT_ELBOW)
        reparentTo(self.mp_nodes[lms.RIGHT_ELBOW],
                   self.mp_nodes[lms.RIGHT_SHOULDER])
        setLandmark(lms.RIGHT_WRIST)
        reparentTo(self.mp_nodes[lms.RIGHT_WRIST],
                   self.mp_nodes[lms.RIGHT_ELBOW])

        setLandmark(lms.LEFT_SHOULDER)
        reparentTo(self.mp_nodes[lms.LEFT_SHOULDER], self.mp_nodes[lms.NOSE])
        setLandmark(lms.LEFT_ELBOW)
        reparentTo(self.mp_nodes[lms.LEFT_ELBOW],
                   self.mp_nodes[lms.LEFT_SHOULDER])
        setLandmark(lms.LEFT_WRIST)
        reparentTo(self.mp_nodes[lms.LEFT_WRIST],
                   self.mp_nodes[lms.LEFT_ELBOW])

        setLandmark(lms.RIGHT_HIP)
        reparentTo(self.mp_nodes[lms.RIGHT_HIP],
                   self.mp_nodes[lms.RIGHT_SHOULDER])
        setLandmark(lms.RIGHT_KNEE)
        reparentTo(self.mp_nodes[lms.RIGHT_KNEE], self.mp_nodes[lms.RIGHT_HIP])
        setLandmark(lms.RIGHT_ANKLE)
        reparentTo(self.mp_nodes[lms.RIGHT_ANKLE],
                   self.mp_nodes[lms.RIGHT_KNEE])

        setLandmark(lms.LEFT_HIP)
        reparentTo(self.mp_nodes[lms.LEFT_HIP],
                   self.mp_nodes[lms.LEFT_SHOULDER])
        setLandmark(lms.LEFT_KNEE)
        reparentTo(self.mp_nodes[lms.LEFT_KNEE], self.mp_nodes[lms.LEFT_HIP])
        setLandmark(lms.LEFT_ANKLE)
        reparentTo(self.mp_nodes[lms.LEFT_ANKLE],
                   self.mp_nodes[lms.LEFT_KNEE])

        self.ikchains = {}

        for lm in lms:
            if lm in IK_CHAIN_INFOS:
                self.ikchains[lm] = initIKChain(lm, IK_CHAIN_INFOS[lm])

    def setMPPose(self, pose, landmark: mp_pose.PoseLandmark):
        if landmark not in self.mp_nodes:
            return

        pos = pose.pose_world_landmarks.landmark[landmark]
        # print(f"{landmark} : {pos}")
        pos = Vec3(-pos.x * LANDMARK_SCALE, pos.z *
                   LANDMARK_SCALE, -pos.y * LANDMARK_SCALE)
        # q = Quat()
        # q.setHpr(Vec3(0, -90, 0))
        # pos = q.xform(pos)
        self.mp_nodes[landmark].setFluidPos(pos)
        if landmark in self.ikchains:
            self.ikchains[landmark][0].update_ik()
        # mp_node = self.mp_nodes[landmark]
        # if landmark in joint_mapping:
        #     mat = mp_node.getMat()
        #     self.joints[joint_mapping[landmark]].setPos(
        #         mp_node.getPos())

    def poseEstimation(self, time):
        # nosepos = self.mp_nodes[mp_pose.PoseLandmark.NOSE].getPos()
        # self.camera.setPos(nosepos.x, nosepos.y - 50, nosepos.z)
        # if time.frame % 2 == 0:
        #     return Task.cont

        ret, frame = self.video.read()
        image_height, image_width, _ = frame.shape
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_world_landmarks:
            return Task.cont

        wlandmarks = results.pose_world_landmarks
        # win_width = self.win.getProperties().getXSize()
        # win_height = self.win.getProperties().getYSize()
        landmarks = [[lm, Vec3(-wlandmarks.landmark[lm].x, wlandmarks.landmark[lm].z, -wlandmarks.landmark[lm].y)]
                     for idx, lm in enumerate(mp_pose.PoseLandmark)]
        positions, rotations, scalings = calculate_inverse_kinematics(
            landmarks)

        # print(rotations)
        for landmark in mp_pose.PoseLandmark:
            self.setMPPose(results, landmark)

        # for lm in landmarks:
        #     if lm[0] in self.ikchains:
        #         # q = Quat()
        #         # q.setHpr(Vec3(0, -90, 0))
        #         # pos = q.xform(lm[1] * LANDMARK_SCALE)
        #         self.ikchains[lm[0]][0].update_ik()

        # for position in positions:
        #     if position[0] in joint_mapping:
        #         self.joints[joint_mapping[position[0]]
        #                     ].setFluidPos(position[1])
        # for rotation in rotations:
        #     self.joints[joint_mapping[rotation[0]]].setQuat(rotation[1])

        # print(jointNow)
        # print(right_shoulder, jointNow)
        # distance = jointNow - right_shoulder
        # self.dummy.setFluidPos(0, 10, 0)

        return Task.cont

    # def spinJoint(self, time):
    #     tmp = self.counter / 5
    #
    #     quat = LQuaternionf()
    #     quat.setFromAxisAngle(-tmp, LVector3f(1, 0, 0))
    #     # quat.setFromAxisAngle(tmp, LVector3f(0, 1, 0))
    #     self.dummy.setQuat(quat)
    #     self.counter += 1
    #     if self.counter >= 180:
    #         self.counter = 0
    #
    #     return Task.cont


app = Animate()
app.run()
