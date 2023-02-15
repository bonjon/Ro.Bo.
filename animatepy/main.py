#!/usr/bin/env python3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from enum import Enum
from panda3d.core import Material, DirectionalLight, Vec3, Vec4, Mat4, Mat3, LPoint3, ClientBase, NodePath, LineSegs, GeomVertexWriter, PointLight, AmbientLight, Quat, CharacterJointBundle, CharacterJoint, PartBundle
import math

from exts.ik.CCDIK.ik_actor import IKActor, IKChain

import numpy as np
import cv2
import mediapipe as mp
import argparse
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class CustomLandmark(Enum):
    ShoulderCenter = 100,
    HipCenter = 101,
    Offset = 102,


class ModelCustomizations(Enum):
    No = 0,
    Angel = 1,
    Devil = 2,
    Pirate = 3


LANDMARK_SCALE = 10


def center_point(a: Vec3, b: Vec3):
    return (a + b) / 2


def convert_lm_to_vec3(pos):
    return Vec3(pos.x * LANDMARK_SCALE, pos.z *
                LANDMARK_SCALE / 2, -pos.y * LANDMARK_SCALE)


def transform_to_hip_origin(world_landmarks):
    hip: Vec3 = center_point(world_landmarks[23][1], world_landmarks[24][1])
    world_landmarks = [[idx, lm - hip] for idx, lm in world_landmarks]
    world_landmarks.append([CustomLandmark.Offset, hip])
    return world_landmarks


def find_parent(root: PartBundle, to_find):
    print(root.getChildren(), to_find)
    levels = root.getChildren()
    while levels:
        next_level = []

        for node in levels:
            if to_find in [x.getName() for x in node.getChildren()]:
                return node

            next_level += node.getChildren()
        levels = next_level


def generate_ik_chain_info(model: Actor) -> dict:
    right_arm = [j.getName() for j in model.getJoints(None, "right_arm_*")]
    left_arm = [j.getName() for j in model.getJoints(None, "left_arm_*")]
    right_leg = [j.getName() for j in model.getJoints(None, "right_leg_*")]
    left_leg = [j.getName() for j in model.getJoints(None, "left_leg_*")]
    head = [j.getName() for j in model.getJoints(None, "head_*")]

    part_bundle = model.getPartBundle("modelRoot")

    print(model.listJoints())

    left_arm_parent = find_parent(part_bundle, "left_arm_0")
    right_arm_parent = find_parent(part_bundle, "right_arm_0")
    hip = find_parent(part_bundle, "left_leg_0")

    print("Hip:", hip.getName())

    right_arm_constraints = [
        {
            'type': 1,
            'joint': 'right_arm_0',
            'unit': -Vec3.unit_z(),
            'min_angle': -np.pi * 0.5,
            'max_angle': np.pi * 0.5
        },
        # {
        #     'type': 1,
        #     'joint': 'right_arm_1',
        #     'unit': -Vec3.unit_z(),
        #     'min_angle': -np.pi * 0.25,
        #     'max_angle': np.pi * 0.5
        # },
    ]

    if len(right_arm_constraints) < len(right_arm):
        right_arm_constraints = right_arm_constraints[:len(right_arm) - 1]

    left_arm_constraints = [
        {
            'type': 1,
            'joint': 'left_arm_0',
            'unit': Vec3.unit_z(),
            'min_angle': -np.pi * 0.5,
            'max_angle': np.pi * 0.5
        },
        # {
        #     'type': 1,
        #     'joint': 'left_arm_1',
        #     'unit': Vec3.unit_z(),
        #     'min_angle': -np.pi * 0.25,
        #     'max_angle': np.pi * 0.5
        # },
    ]

    if len(left_arm_constraints) < len(left_arm):
        left_arm_constraints = left_arm_constraints[:len(left_arm) - 1]

    right_leg_constraints = [
        {
            'type': 1,
            'joint': 'right_leg_0',
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'right_leg_1',
            'unit': Vec3.unit_z(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': right_leg[math.ceil((len(right_leg) - 1) / 2)],
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.20,
            'max_angle': np.pi * 0.20,
        },
        # The ankle, moves just left or right
        {
            'type': 1,
            'joint': right_leg[-2],
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.25,
        },
        {
            'type': 1,
            'joint': right_leg[-1],
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.1,
            'max_angle': np.pi * 0.1,
        },
        # Other joints that we may want to limit, but are not essential
        {
            'type': 1,
            'joint': 'right_leg_2',
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'right_leg_3',
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'right_leg_4',
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'right_leg_5',
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
    ]

    if len(right_leg) < len(right_leg_constraints):
        right_leg_constraints = right_leg_constraints[:len(right_leg) - 1]

    left_leg_constraints = [
        {
            'type': 1,
            'joint': 'left_leg_0',
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'left_leg_1',
            'unit': Vec3.unit_z(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': left_leg[math.ceil((len(left_leg) - 1) / 2)],
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.20,
            'max_angle': np.pi * 0.20,
        },
        # The ankle, moves just left or right
        {
            'type': 1,
            'joint': left_leg[-2],
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.25,
        },
        {
            'type': 1,
            'joint': left_leg[-1],
            'unit': Vec3.unit_y(),
            'min_angle': -np.pi * 0.1,
            'max_angle': np.pi * 0.1,
        },
    ]

    print(right_leg[:-2])
    print(right_leg[-2:])

    if len(left_leg) < len(left_leg_constraints):
        left_leg_constraints = left_leg_constraints[:len(left_leg) - 1]

    head_constraints = [
        {
            'type': 1,
            'joint': 'head_0',
            'unit': Vec3.unit_z(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        {
            'type': 1,
            'joint': 'head_1',
            'unit': Vec3.unit_x(),
            'min_angle': -np.pi * 0.05,
            'max_angle': np.pi * 0.05,
        },
        # {
        #     'type': 1,
        #     'joint': 'head_2',
        #     'unit': Vec3.unit_x(),
        #     'min_angle': -np.pi * 0.1,
        #     'max_angle': np.pi * 0.1,
        # }
    ]

    if len(head) < len(head_constraints):
        head_constraints = head_constraints[:len(head) - 1]

    return {
        # CustomLandmark.HipCenter: {
        #     'joints': ['root'],
        #     'constraints': [],
        # },
        #
        # mp_pose.PoseLandmark.LEFT_SHOULDER: {
        #     'joints': [left_arm[0]],
        #     'constraints': [
        #         # {
        #         #     'type': 1,
        #         #     'joint': left_arm[0],
        #         #     'unit': Vec3.unit_x(),
        #         #     'min_angle': -np.pi * 0.25,
        #         #     'max_angle': np.pi * 0.1
        #         # }
        #     ],
        # },

        mp_pose.PoseLandmark.LEFT_ELBOW: {
            'joints': left_arm[:-2],
            'constraints': [
                # {
                #     'type': 1,
                #     'joint': left_arm[0],
                #     'unit': Vec3.unit_z(),
                #     'min_angle': -np.pi * 0.25,
                #     'max_angle': np.pi * 0.5
                # }
            ],
        },

        mp_pose.PoseLandmark.LEFT_WRIST: {
            'joints': left_arm[-3:],
            'constraints': []
        },

        # mp_pose.PoseLandmark.RIGHT_SHOULDER: {
        #     'joints': [right_arm[0]],
        #     'constraints': [
        #         # {
        #         #     'type': 1,
        #         #     'joint': right_arm_parent.getName(),
        #         #     'unit': Vec3.unit_x(),
        #         #     'min_angle': -np.pi * 0.25,
        #         #     'max_angle': np.pi * 0.1
        #         # }
        #     ],
        # },

        mp_pose.PoseLandmark.RIGHT_ELBOW: {
            'joints': right_arm[:-2],
            'constraints': right_arm_constraints,
        },

        mp_pose.PoseLandmark.RIGHT_WRIST: {
            'joints': right_arm[-3:],
            'constraints': [],
        },

        mp_pose.PoseLandmark.LEFT_KNEE: {
            'joints': left_leg[:-2],
            'constraints': []
        },

        mp_pose.PoseLandmark.LEFT_ANKLE: {
            'joints': left_leg[-3:],
            'constraints': []
        },


        mp_pose.PoseLandmark.RIGHT_KNEE: {
            'joints': right_leg[:-2],
            'constraints': []
        },


        mp_pose.PoseLandmark.RIGHT_ANKLE: {
            'joints': right_leg[-3:],
            'constraints': []
        },

        mp_pose.PoseLandmark.NOSE: {
            'joints': head,
            'constraints': head_constraints,
        },
    }


class Animate(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.dlight = PointLight('my dlight')
        self.dlnp = self.render.attachNewNode(self.dlight)
        self.dlnp.setPos(0, -100, 20)
        self.render.setLight(self.dlnp)
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        self.rootNode = self.render.attachNewNode("Torso")
        self.mp_nodes_root = self.render.attachNewNode("MPRoot")
        self.mp_nodes_root.hide()
        self.rootNode.setScale(8, 8, 8)
        self.rootNode.setHpr(0, 180, 0)
        parser = argparse.ArgumentParser()
        parser.add_argument('model')
        args = parser.parse_args()
        self.model = Actor(args.model)
        # self.model.setColor((1, 1, 1, 1))
        self.ikmodel = IKActor(self.model)
        self.ikmodel.reparent_to(self.rootNode)
        print(self.ikmodel.actor)
        self.ik_chain_info = generate_ik_chain_info(self.model)

        self.camera.setPos(0, 0, 100)
        self.material = Material()
        self.material.setShininess(5.0)  # Make this material shiny
        self.material.setAmbient((1, 0, 0, 1))  # Make this material blue
        self.model.setMaterial(self.material)
        self.video = cv2.VideoCapture(0)

        self.mp_nodes = {}
        self.init_MP_landmarks()

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.7)

        self.mode = ModelCustomizations.No

        self.setup_model_customizations()

        self.accept('c', self.change_mode)

        # self.taskMgr.add(self.spinJoint, "SpinJoint")
        self.taskMgr.add(self.poseEstimation, "Pose Estimation")

    def setup_model_customizations(self):
        heads = [j.getName() for j in self.model.getJoints(None, "head_*")]

        head = self.ikchains[mp_pose.PoseLandmark.NOSE].get_bone(
            heads[-1])

        self.horn = self.loader.loadModel("horns.glb")
        self.horn.setScale(0.04, 0.04, 0.04)
        self.horn.setPos(0, 0, 0)
        self.horn.setHpr(0, 180, 180)

        self.horn.reparentTo(head.control_node)

        self.pirate_hat = self.loader.loadModel("pirate_hat.glb")
        self.pirate_hat.setScale(0.05, 0.05, 0.05)
        self.pirate_hat.setPos(0, 0.1, 0.1)
        self.pirate_hat.setHpr(0, 180, 180)
        self.pirate_hat.reparentTo(head.control_node)

        self.halo = self.loader.loadModel("halo.glb")
        self.halo.setScale(0.05, 0.05, 0.05)
        self.halo.setPos(0, 0.05, 0.1)
        self.halo.setHpr(0, 180, 180)
        self.halo.reparentTo(head.control_node)

        self.horn.hide()
        self.pirate_hat.hide()
        self.halo.hide()

    def change_mode(self):
        self.horn.hide()
        self.halo.hide()
        self.pirate_hat.hide()

        if self.mode == ModelCustomizations.Devil:
            self.horn.show()
            self.mode = ModelCustomizations.Angel
        elif self.mode == ModelCustomizations.Angel:
            self.halo.show()
            self.mode = ModelCustomizations.Pirate
        elif self.mode == ModelCustomizations.Pirate:
            self.pirate_hat.show()
            self.mode = ModelCustomizations.No
        else:
            self.mode = ModelCustomizations.Devil

    def init_MP_landmarks(self):
        self.joints = {}

        lms = mp_pose.PoseLandmark

        def setLandmark(lm):
            material = Material()
            material.setShininess(5.0)  # Make this material shiny
            material.setAmbient((0, 0, 1, 1))  # Make this material blue
            self.mp_nodes[lm] = self.loader.loadModel('sphere.glb')
            self.mp_nodes[lm].setMaterial(material)
            self.mp_nodes[lm].setScale(0.25)
            self.mp_nodes[lm].reparentTo(self.mp_nodes_root)

        def initIKChain(lm, ikInfo):
            chain = self.ikmodel.create_ik_chain(ikInfo['joints'])
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
            # chain.debug_display()
            return chain

        setLandmark(lms.NOSE)
        setLandmark(lms.RIGHT_SHOULDER)
        setLandmark(lms.RIGHT_ELBOW)
        setLandmark(lms.RIGHT_WRIST)
        setLandmark(lms.LEFT_SHOULDER)
        setLandmark(lms.LEFT_ELBOW)
        setLandmark(lms.LEFT_WRIST)
        setLandmark(lms.RIGHT_HIP)
        setLandmark(lms.RIGHT_KNEE)
        setLandmark(lms.RIGHT_ANKLE)
        setLandmark(lms.LEFT_HIP)
        setLandmark(lms.LEFT_KNEE)
        setLandmark(lms.LEFT_ANKLE)
        setLandmark(lms.RIGHT_ANKLE)

        setLandmark(CustomLandmark.ShoulderCenter)
        # setLandmark(CustomLandmark.HipCenter)

        self.ikchains = {}

        for lm in lms:
            if lm in self.ik_chain_info:
                self.ikchains[lm] = initIKChain(lm, self.ik_chain_info[lm])

        # self.ikchains[CustomLandmark.HipCenter] = initIKChain(
        #     CustomLandmark.HipCenter, self.ik_chain_info[CustomLandmark.HipCenter])

    def setMPPose(self, pose, landmark: mp_pose.PoseLandmark):
        if landmark == CustomLandmark.HipCenter:
            lm_0 = mp_pose.PoseLandmark.LEFT_HIP
            lm_1 = mp_pose.PoseLandmark.RIGHT_HIP
            pos_0 = pose.pose_world_landmarks.landmark[lm_0]
            pos_1 = pose.pose_world_landmarks.landmark[lm_1]

            pos_0 = convert_lm_to_vec3(pos_0)
            pos_1 = convert_lm_to_vec3(pos_1)
            hip = center_point(pos_0, pos_1)

            self.mp_nodes[landmark].setFluidPos(hip)
            if landmark in self.ikchains:
                self.ikchains[landmark].update_ik()

            return

        if landmark not in self.mp_nodes:
            return

        pos = pose.pose_world_landmarks.landmark[landmark]
        if pos.visibility <= 0.60:
            return

        # print(f"{landmark} : {pos}")
        pos = convert_lm_to_vec3(pos)
        self.mp_nodes[landmark].setFluidPos(pos)
        if landmark in self.ikchains:
            self.ikchains[landmark].update_ik()

    def poseEstimation(self, time):
        ret, frame = self.video.read()
        image_height, image_width, _ = frame.shape
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_world_landmarks:
            return Task.cont

        for landmark in mp_pose.PoseLandmark:
            self.setMPPose(results, landmark)

        # self.setMPPose(results, CustomLandmark.HipCenter)

        return Task.cont


app = Animate()
app.run()
