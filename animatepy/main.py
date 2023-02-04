#!/usr/bin/env python3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from panda3d.core import Material, DirectionalLight, Vec3, Mat4,  LPoint3, ClientBase, NodePath, LineSegs, GeomVertexWriter, PointLight, AmbientLight, Quat

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def center_point(a: Vec3, b: Vec3):
    return (a + b) / 2


def transform_to_hip_origin(world_landmarks):
    hip: Vec3 = center_point(world_landmarks[23][1], world_landmarks[24][1])
    return [[idx, lm - hip] for idx, lm in world_landmarks]


def rotate_towards(source: Vec3, dest: Vec3, axis: Vec3, up: Vec3) -> Quat:
    direction: Vec3 = dest - source
    direction.normalize()
    up: Vec3 = direction.up()
    up.normalize()
    axis: Vec3 = -direction.left()
    axis.normalize()

    rotation_mat: Mat4 = Mat4()
    rotation_mat.setRow(0, axis)
    rotation_mat.setRow(1, up)
    rotation_mat.setRow(2, direction)
    rotation_mat.setRow(3, Vec3(1, 1, 1))
    translation_mat: Mat4 = Mat4()
    translation_mat.setCol(3, -source)
    look_at = rotation_mat * translation_mat

    quat = Quat()
    quat.setFromMatrix(look_at)

    return quat


def limb_rotations(world_landmarks):
    res = []
    lms = mp_pose.PoseLandmark
    world_landmarks = transform_to_hip_origin(world_landmarks)

    def calc_chain_rotations(data):
        rotations = []
        for i in range(1, len(data)):
            # print(data[i][1], data[i - 1][1])
            # quart = rotate_towards(
            #     data[i][1], data[i - 1][1], '-Y', 'Z')
            # source = Vec3(data[i][1].x, data[i][1].y, data[i][1].z)
            # dest = Vec3(data[i - 1][1].x, data[i - 1][1].y, data[i - 1][1].z)
            quart = rotate_towards(
                data[i][1], data[i - 1][1], Vec3(), Vec3())
            rotations.append([data[i - 1][0], quart])
        return rotations

    left_arm = [world_landmarks[lms.LEFT_SHOULDER],
                world_landmarks[lms.LEFT_ELBOW], world_landmarks[lms.LEFT_WRIST]]
    right_arm = [world_landmarks[lms.RIGHT_SHOULDER],
                 world_landmarks[lms.RIGHT_ELBOW], world_landmarks[lms.RIGHT_WRIST]]
    for limbs in [left_arm, right_arm]:
        res += calc_chain_rotations(limbs)
    return res


def calculate_inverse_kinematics(world_landmarks):
    joint_rotations = []
    joint_scaling = {}

    limbs = limb_rotations(world_landmarks)
    joint_rotations += limbs

    print(joint_rotations)
    return joint_rotations, joint_scaling


# import mediapipe

# def convert_coordinate_system(p_from: LPoint3, frame):

joint_mapping = {
    mp_pose.PoseLandmark.NOSE: 'joint_11',
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 'joint_15',
    mp_pose.PoseLandmark.RIGHT_ELBOW: 'joint_14',
    mp_pose.PoseLandmark.RIGHT_WRIST: 'joint_12',

    mp_pose.PoseLandmark.LEFT_SHOULDER: 'joint_3',
    mp_pose.PoseLandmark.LEFT_ELBOW: 'joint_2',
    mp_pose.PoseLandmark.LEFT_WRIST: 'joint_0',

    mp_pose.PoseLandmark.LEFT_HIP: 'root_dup_0',
    mp_pose.PoseLandmark.LEFT_KNEE: 'joint_6',
    mp_pose.PoseLandmark.LEFT_ANKLE: 'joint_5',

    mp_pose.PoseLandmark.RIGHT_HIP: 'root_dup_2',
    mp_pose.PoseLandmark.RIGHT_KNEE: 'joint_18',
    mp_pose.PoseLandmark.RIGHT_ANKLE: 'joint_17',


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
        self.render.setLight(alnp)
        self.model = Actor('model.glb')

        print(self.model.listJoints())

        self.model.reparentTo(self.render)
        self.model.setScale(0.75, 0.75, 0.75)
        self.model.setPos(0, 0, -5)
        self.model.setHpr(0, 90, 0)
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
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)

        self.counter = 0

        # self.taskMgr.add(self.spinJoint, "SpinJoint")
        self.taskMgr.add(self.poseEstimation, "Pose Estimation")

    def initMPLandmarks(self):
        self.joints = {}

        for joint in self.model.getJoints():
            self.joints[joint.getName()] = self.model.controlJoint(
                None, "modelRoot", joint.getName())

        lms = mp_pose.PoseLandmark

        def setLandmark(lm):
            material = Material()
            material.setShininess(5.0)  # Make this material shiny
            material.setAmbient((0, 0, 1, 1))  # Make this material blue
            self.mp_nodes[lm] = self.loader.loadModel('sphere.glb')
            self.mp_nodes[lm].setMaterial(material)

        def reparentTo(node, parent):
            node.setScale(0.25)
            node.reparentTo(self.render)
            # node.reparentTo(parent)

        setLandmark(lms.NOSE)
        self.mp_nodes[lms.NOSE].setScale(0.25)
        self.mp_nodes[lms.NOSE].reparentTo(self.render)
        self.mp_nodes[lms.NOSE].setHpr(0, 0, 0)

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

        # print(self.mp_nodes)

    def setMPPose(self, pose, landmark: mp_pose.PoseLandmark):
        SCALE = 5
        if landmark not in self.mp_nodes:
            return

        pos = pose.pose_world_landmarks.landmark[landmark]
        # print(f"{landmark} : {pos}")
        self.mp_nodes[landmark].setFluidPos(
            -pos.x, pos.z, -pos.y)
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
        rotations, scalings = calculate_inverse_kinematics(landmarks)

        # print(rotations)
        for landmark in mp_pose.PoseLandmark:
            self.setMPPose(results, landmark)

        for rotation in rotations:
            self.joints[joint_mapping[rotation[0]]].setQuat(rotation[1])

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
