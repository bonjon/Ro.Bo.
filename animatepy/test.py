#!/usr/bin/env python3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from enum import Enum
from panda3d.core import Material, DirectionalLight, Vec3, Vec4, Mat4, Mat3, LPoint3, ClientBase, NodePath, LineSegs, GeomVertexWriter, PointLight, AmbientLight, Quat

from exts.ik.CCDIK.ik_actor import IKActor, IKChain

import numpy as np


def create_axes(size, bothways=False, thickness=1):

    lines = LineSegs()
    lines.set_thickness(thickness)

    lines.set_color(1, 0.1, 0.1, 0.1)
    if bothways:
        lines.move_to(-size, 0, 0)
    else:
        lines.move_to(0, 0, 0)
    lines.draw_to(size, 0, 0)

    lines.set_color(0.1, 1, 0.1, 0.1)
    if bothways:
        lines.move_to(0, -size, 0)
    else:
        lines.move_to(0, 0, 0)
    lines.draw_to(0, size, 0)

    lines.set_color(0.1, 0.1, 1, 0.1)
    if bothways:
        lines.move_to(0, 0, -size)
    else:
        lines.move_to(0, 0, 0)
    lines.draw_to(0, 0, size)

    geom = lines.create()
    return geom


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

        self.root_node = self.render.attachNewNode("Torso")
        geom = create_axes(0.3)
        self.root_node.attach_new_node(geom)
        self.target__node = self.render.attach_new_node("Walk target")

        self.model = Actor('giua.glb')
        # self.model = Actor('person.glb')
        self.ikmodel = IKActor(self.model)
        print(self.model.list_joints())
        self.ikmodel.reparent_to(self.root_node)

        self.ik_chain_left_arm = self.ikmodel.create_ik_chain(
            ['joint_6', 'joint_8', 'joint_7', 'joint_1'])
        # self.ik_chain_left_arm = self.ikmodel.create_ik_chain(
        #     ['Shoulder.L', 'UpperArm.L', 'LowerArm.L', 'Hand.L'])
        self.ik_chain_left_arm.debug_display(line_length=0.1)


app = Animate()
app.run()
