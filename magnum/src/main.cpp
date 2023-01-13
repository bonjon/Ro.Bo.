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
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/SkinData.h>
#include <MagnumPlugins/AnyImageImporter/AnyImageImporter.h>
#include <MagnumPlugins/GltfImporter/GltfImporter.h>

using namespace Magnum;
using namespace Magnum::Math::Literals;

class MyApplication : public Platform::Application {
public:
  explicit MyApplication(const Arguments &arguments);

private:
  void drawEvent() override;

  Shaders::PhongGL m_shader;
  Corrade::Containers::Optional<Trade::SkinData3D> m_skin;
  GL::Mesh m_mesh;
  Color3 m_color;
  Matrix4 m_transformation;
  Matrix4 m_projection;
  Containers::Array<Matrix4> absoluteTransformations{};
  Math::Deg<float> m_move_joint;
};

MyApplication::MyApplication(const Arguments &arguments)
    : Platform::Application{arguments} {

  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

  PluginManager::Manager<Trade::AbstractImporter> manager;

  Trade::GltfImporter importer;

  importer.openFile("smith.glb");
  Debug{} << "Scene count: " << importer.sceneCount();
  auto scene = importer.scene(0);
  Debug{} << "Mesh count: " << importer.meshCount();
  auto mesh = importer.mesh(0);

  Debug{} << "Skins: " << importer.skin3DCount();
  m_skin = importer.skin3D(0);

  Debug{} << "Joints: " << m_skin->joints();

  for (int i = 0; i < mesh->attributeCount(); ++i) {
    Debug{} << mesh->attributeName(i);
  }

  m_mesh = MeshTools::compile(*mesh);
  Containers::Pair<UnsignedInt, UnsignedInt> meshPerVertexJointCount =
      MeshTools::compiledPerVertexJointCount(*mesh);

  m_shader = Shaders::PhongGL{Shaders::PhongGL::Configuration{}.setJointCount(
      m_skin->joints().size(), meshPerVertexJointCount.first(),
      meshPerVertexJointCount.second())};

  absoluteTransformations = Containers::Array<Matrix4>{m_skin->joints().size()};

  const auto weights = mesh->weightsAsArray();
  const auto joints = m_skin->joints();

  Debug{} << "Weights:" << weights.size() << "Joints:" << joints.size()
          << "W/J:" << weights.size() / joints.size();

  m_transformation =
      Matrix4::rotationX(10.0_degf) * Matrix4::rotationY(-25.0_degf);
  m_projection =
      Matrix4::perspectiveProjection(
          90.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f) *
      Matrix4::translation({0.0f, -5, -15});
  m_color = Color3::fromSrgb({0.6, 0.7, 0.7});
}

void MyApplication::drawEvent() {
  GL::defaultFramebuffer.clearColor({0.25f, 0.1f, 0.25f});
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Depth);

  absoluteTransformations[1] =
      absoluteTransformations[1].rotationZ(m_move_joint);
  // absoluteTransformations[24] =
  //     absoluteTransformations[24]
  //         .rotationY(m_move_joint / 5)
  //         .translation({float(m_move_joint) / 50, 0, float(m_move_joint) /
  //         25});

  Containers::Array<Matrix4> jointTransformations{m_skin->joints().size()};
  for (std::size_t i = 0; i < jointTransformations.size(); ++i) {
    jointTransformations[i] =
        absoluteTransformations[i]; //* m_skin->inverseBindMatrices()[i];
  }

  m_shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
      .setDiffuseColor(m_color)
      .setAmbientColor(Color3::fromHsv({m_color.hue(), 1.0f, 0.3f}))
      .setTransformationMatrix(m_transformation)
      .setNormalMatrix(m_transformation.normalMatrix())
      .setProjectionMatrix(m_projection)
      .setJointMatrices(jointTransformations)
      .draw(m_mesh);

  swapBuffers();
  redraw();
  if (m_move_joint <= -45.0_degf) {
    m_move_joint = 0.0_degf;
  } else {
    m_move_joint -= 1.0_degf;
  }
}

MAGNUM_APPLICATION_MAIN(MyApplication)
