#include "Magnum/GL/Mesh.h"
#include "Magnum/GL/Shader.h"
#include "Magnum/Magnum.h"
#include "Magnum/Math/Angle.h"
#include "Magnum/MeshTools/Compile.h"
#include "Magnum/Shaders/PhongGL.h"
#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/TranslationRotationScalingTransformation3D.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/SkinData.h>
#include <MagnumPlugins/AnyImageImporter/AnyImageImporter.h>
#include <MagnumPlugins/GltfImporter/GltfImporter.h>

#include <memory>

using namespace Magnum;
using namespace Magnum::Math::Literals;

using Object3D =
    SceneGraph::Object<SceneGraph::TranslationRotationScalingTransformation3D>;

using Scene3D =
    SceneGraph::Scene<SceneGraph::TranslationRotationScalingTransformation3D>;

class JointDrawable : public SceneGraph::Drawable3D {
public:
  explicit JointDrawable(Object3D &object, const Matrix4 &inverseBindMatrix,
                         Matrix4 &jointMatrix,
                         SceneGraph::DrawableGroup3D &group)
      : SceneGraph::Drawable3D{object, &group},
        _inverseBindMatrix{inverseBindMatrix},
        /* GCC 4.8 can't handle {} here */
        _jointMatrix(jointMatrix) {}

private:
  void draw(const Matrix4 &transformationMatrix,
            SceneGraph::Camera3D &) override {
    _jointMatrix = transformationMatrix * _inverseBindMatrix;
  }

  Matrix4 _inverseBindMatrix;
  Matrix4 &_jointMatrix;
};

class MyApplication : public Platform::Application {
public:
  explicit MyApplication(const Arguments &arguments);

private:
  void drawEvent() override;
  void calculateSkinning(const Trade::SceneData &scene,
                         const Trade::SkinData3D &newSkinning);

  Shaders::PhongGL m_shader;
  Corrade::Containers::Optional<Trade::SkinData3D> m_skin;
  Object3D m_meshObject;
  GL::Mesh m_mesh;
  Color3 m_color;
  Matrix4 m_transformation;
  Matrix4 m_projection;
  Containers::Array<Matrix4> m_skinJointMatrices;
  Math::Deg<float> m_move_joint;

  Scene3D m_scene;
  std::unique_ptr<SceneGraph::Camera3D> m_rootCamera;
  std::unique_ptr<Object3D> m_rootCameraObject;
  SceneGraph::DrawableGroup3D m_jointDrawables;
  Containers::Array<Object3D> m_objects;
};

MyApplication::MyApplication(const Arguments &arguments)
    : Platform::Application{arguments} {

  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

  PluginManager::Manager<Trade::AbstractImporter> manager;

  Trade::GltfImporter importer;

  importer.openFile("model.glb");
  Debug{} << "Scene count: " << importer.sceneCount();
  auto scene = importer.scene(importer.defaultScene());
  Debug{} << "Mapping bound:" << scene->mappingBound();
  Debug{} << "Mesh count: " << importer.meshCount();
  auto mesh = importer.mesh(0);
  const Containers::Array<Containers::Pair<UnsignedInt, Int>> parents =
      scene->parentsAsArray();

  m_meshObject.setParent(&m_scene);

  m_rootCameraObject = std::make_unique<Object3D>(&m_scene);
  m_rootCamera = std::make_unique<SceneGraph::Camera3D>(*m_rootCameraObject);

  Debug{} << "Skins: " << importer.skin3DCount();
  m_skin = importer.skin3D(0);
  if (m_skin) {
    calculateSkinning(*scene, *m_skin);
  }

  Debug{} << "Animations: " << importer.animationCount();

  Debug{} << "Joints: " << m_skin->joints();
  Debug{} << "Inverse Bind: "
          << m_skin->inverseBindMatrices()[m_skin->joints()[0]];

  for (int i = 0; i < mesh->attributeCount(); ++i) {
    Debug{} << mesh->attributeName(i);
  }

  m_mesh = MeshTools::compile(*mesh);
  Containers::Pair<UnsignedInt, UnsignedInt> meshPerVertexJointCount =
      MeshTools::compiledPerVertexJointCount(*mesh);

  m_shader = Shaders::PhongGL{Shaders::PhongGL::Configuration{}.setJointCount(
      m_skin->joints().size(), meshPerVertexJointCount.first(),
      meshPerVertexJointCount.second())};

  const auto weights = mesh->weightsAsArray();
  const auto joints = m_skin->joints();

  Debug{} << "Weights:" << weights.size() << "Joints:" << joints.size()
          << "W/J:" << weights.size() / joints.size();

  m_transformation =
      Matrix4::rotationX(10.0_degf) * Matrix4::rotationY(-25.0_degf);
  // Matrix4::scaling({10, 10, 10});
  m_projection =
      Matrix4::perspectiveProjection(
          90.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f) *
      Matrix4::translation({0.0f, -6, -15});
  m_color = Color3::fromSrgb({0.6, 0.7, 0.7});
}

void MyApplication::calculateSkinning(const Trade::SceneData &scene,
                                      const Trade::SkinData3D &skinData) {
  m_objects =
      Containers::Array<Object3D>(ValueInit, std::size_t{scene.mappingBound()});

  const Containers::Array<Containers::Pair<UnsignedInt, Int>> parents =
      scene.parentsAsArray();
  for (const Containers::Pair<UnsignedInt, Int> &parent : parents) {
    const UnsignedInt objectId = parent.first();
    m_objects[objectId].setParent(
        parent.second() == -1 ? &m_scene : &m_objects[parent.second()]);

    // if (parent.second() != -1)
    //   ++_data->objects[parent.second()].childCount;
    // _data->objects[objectId].object = new Object3D{};
    // _data->objects[objectId].type = "empty";
    // _data->objects[objectId].name = importer.objectName(objectId);
    // if (_data->objects[objectId].name.empty())
    //   _data->objects[objectId].name =
    //       Utility::formatString("object #{}", objectId);
  }

  for (const Containers::Pair<UnsignedInt, Matrix4> &transformation :
       scene.transformations3DAsArray()) {
    Object3D &object = m_objects[transformation.first()];
    object.setTransformation(transformation.second());
  }

  const auto &joints = skinData.joints();
  m_skinJointMatrices = Containers::Array<Matrix4>{NoInit, joints.size()};
  for (int i = 0; i < joints.size(); ++i) {
    const auto objectId = joints[i];

    // What an hack, holy shit
    new JointDrawable{m_objects[objectId], skinData.inverseBindMatrices()[i],
                      m_skinJointMatrices[i], m_jointDrawables};
  }
}

void MyApplication::drawEvent() {
  GL::defaultFramebuffer.clearColor({0.25f, 0.1f, 0.25f});
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Depth);

  m_objects[23].setRotation(Quaternion::rotation(m_move_joint, {0, 1, 0}));

  // m_skinJointMatrices[3] = m_skinJointMatrices[3].rotationZ(m_move_joint *
  // 10); const auto max_joints = m_skinJointMatrices.size();
  // m_skinJointMatrices[13] =
  //     m_skinJointMatrices[13]
  //         .rotationY(m_move_joint / 5)
  //         .translation({float(m_move_joint) / 50, 0, float(m_move_joint) /
  //         25});

  // Containers::Array<Matrix4> jointTransformations{NoInit,
  //                                                 m_skin->joints().size()};

  // m_rootCameraObject->rotate(Quaternion::rotation(1.0_degf, {0, 1, 0}));
  m_rootCamera->draw(m_jointDrawables);

  m_shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
      .setDiffuseColor(m_color)
      .setAmbientColor(Color3::fromHsv({m_color.hue(), 1.0f, 0.3f}))
      .setTransformationMatrix(m_transformation)
      .setNormalMatrix(m_transformation.normalMatrix())
      .setProjectionMatrix(m_projection)
      .setJointMatrices(m_skinJointMatrices)
      .draw(m_mesh);

  if (m_move_joint <= -45.0_degf) {
    m_move_joint = 0.0_degf;
  } else {
    m_move_joint -= 1.0_degf;
  }

  swapBuffers();
  redraw();
}

MAGNUM_APPLICATION_MAIN(MyApplication)
