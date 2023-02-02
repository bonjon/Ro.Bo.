#!/usr/bin/env python3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.task import Task
from panda3d.core import Material, DirectionalLight, LQuaternionf, LVector3f

# import mediapipe


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.dlight = DirectionalLight('my dlight')
        self.dlnp = self.render.attachNewNode(self.dlight)
        self.render.setLight(self.dlnp)
        self.model = Actor('model.glb')
        print(self.model.listJoints())
        self.dummy = self.model.controlJoint(None, "modelRoot", "joint_10_dup_2")
        print(self.dummy)
        self.model.reparentTo(self.render)
        self.model.setScale(0.75, 0.75, 0.75)
        self.model.setPos(0, 42, -5)
        self.model.setHpr(0, 90, 0)
        self.material = Material()
        self.material.setShininess(5.0) # Make this material shiny
        self.material.setAmbient((0, 0, 1, 1)) # Make this material blue
        self.model.setMaterial(self.material)

        self.counter = 0

        self.taskMgr.add(self.spinJoint, "SpinJoint")

    def spinJoint(self, time):
        tmp = self.counter / 5

        quat = LQuaternionf()
        quat.setFromAxisAngle(-tmp, LVector3f(1, 0, 0))
        # quat.setFromAxisAngle(tmp, LVector3f(0, 1, 0))
        self.dummy.setQuat(quat)
        self.counter += 1
        if self.counter >= 180:
            self.counter = 0

        return Task.cont




app = MyApp()
app.run()
