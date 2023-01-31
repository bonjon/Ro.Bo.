#include "Corrade/Utility/Arguments.h"
#include "Corrade/Utility/FormatStl.h"
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

struct ObjectInfo {
  Object3D object;
  std::string name;
};

struct Person3D {
  Containers::Array<ObjectInfo> objects;
  Containers::Array<Matrix4> jointMatrices;
  Corrade::Containers::Optional<Trade::SkinData3D> skin;
  GL::Mesh mesh;
  SceneGraph::DrawableGroup3D jointDrawables;
  Matrix4 transform;
};

class RoboAnimate : public Platform::Application {
public:
  explicit RoboAnimate(const Arguments &arguments);

private:
  void drawEvent() override;
  Containers::Optional<Person3D>
  calculateSkinning(Trade::GltfImporter &importer);

  Shaders::PhongGL m_shader;

  Containers::Optional<Person3D> m_person;

  // Corrade::Containers::Optional<Trade::SkinData3D> m_skin;
  Object3D m_meshObject;
  Color3 m_color;
  Matrix4 m_projection;
  Containers::Array<Matrix4> m_skinJointMatrices;
  Math::Deg<float> m_move_joint;

  Scene3D m_scene;
  std::unique_ptr<SceneGraph::Camera3D> m_rootCamera;
  std::unique_ptr<Object3D> m_rootCameraObject;
};

RoboAnimate::RoboAnimate(const Arguments &arguments)
    : Platform::Application{arguments,
                            Configuration{}.setTitle("Robo: Animation")} {

  Utility::Arguments argParser;
  // clang-format off
  argParser.addArgument("model").setHelp("model", "The model to load")
           .parse(arguments.argc, arguments.argv);
  // clang-format on

  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

  PluginManager::Manager<Trade::AbstractImporter> manager;

  Trade::GltfImporter importer;

  importer.openFile(argParser.value("model"));

  m_person = calculateSkinning(importer);

  m_rootCameraObject = std::make_unique<Object3D>(&m_scene);
  m_rootCamera = std::make_unique<SceneGraph::Camera3D>(*m_rootCameraObject);

  m_projection =
      Matrix4::perspectiveProjection(
          90.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f) *
      Matrix4::translation({0.0f, -6, -15});
  m_color = Color3::fromSrgb({0.6, 0.7, 0.7});
}

Containers::Optional<Person3D>
RoboAnimate::calculateSkinning(Trade::GltfImporter &importer) {

  auto optScene = importer.scene(importer.defaultScene());
  if (!optScene) {
    return Containers::NullOpt;
  }

  Person3D person;
  person.objects = Containers::Array<ObjectInfo>{
      ValueInit, std::size_t(optScene->mappingBound())};

  person.skin = importer.skin3D(0);

  const Containers::Array<Containers::Pair<UnsignedInt, Int>> parents =
      optScene->parentsAsArray();
  for (const Containers::Pair<UnsignedInt, Int> &parent : parents) {
    const UnsignedInt objectId = parent.first();
    auto &objectInfo = person.objects[objectId];
    objectInfo.name = Utility::formatString("Object #{}", objectId);
    objectInfo.object.setParent(parent.second() == -1
                                    ? &m_scene
                                    : &person.objects[parent.second()].object);
  }

  for (const Containers::Pair<UnsignedInt, Matrix4> &transformation :
       optScene->transformations3DAsArray()) {
    Object3D &object = person.objects[transformation.first()].object;
    object.setTransformation(transformation.second());
  }

  const auto &joints = person.skin->joints();
  person.jointMatrices = Containers::Array<Matrix4>{NoInit, joints.size()};
  for (int i = 0; i < joints.size(); ++i) {
    const auto objectId = joints[i];
    auto &object = person.objects[objectId].object;

    // What an hack, holy shit
    new JointDrawable{object, person.skin->inverseBindMatrices()[i],
                      person.jointMatrices[i], person.jointDrawables};
  }

  auto mesh = importer.mesh(0);
  person.mesh = MeshTools::compile(*mesh);

  Containers::Pair<UnsignedInt, UnsignedInt> meshPerVertexJointCount =
      MeshTools::compiledPerVertexJointCount(*mesh);

  m_shader = Shaders::PhongGL{Shaders::PhongGL::Configuration{}.setJointCount(
      joints.size(), meshPerVertexJointCount.first(),
      meshPerVertexJointCount.second())};

  person.transform =
      Matrix4::rotationX(10.0_degf) * Matrix4::rotationY(-25.0_degf);

  return person;
}

void RoboAnimate::drawEvent() {
  GL::defaultFramebuffer.clearColor({0.25f, 0.1f, 0.25f});
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Depth);

  auto &objects = m_person->objects;

  objects[23].object.setRotation(Quaternion::rotation(m_move_joint, {0, 1, 0}));

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
  m_rootCamera->draw(m_person->jointDrawables);

  m_shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
      .setDiffuseColor(m_color)
      .setAmbientColor(Color3::fromHsv({m_color.hue(), 1.0f, 0.3f}))
      .setTransformationMatrix(m_person->transform)
      .setNormalMatrix(m_person->transform.normalMatrix())
      .setProjectionMatrix(m_projection)
      .setJointMatrices(m_person->jointMatrices)
      .draw(m_person->mesh);

  if (m_move_joint <= -45.0_degf) {
    m_move_joint = 0.0_degf;
  } else {
    m_move_joint -= 1.0_degf;
  }

  swapBuffers();
  redraw();
}

MAGNUM_APPLICATION_MAIN(RoboAnimate)
