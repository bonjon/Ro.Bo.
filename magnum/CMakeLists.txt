cmake_minimum_required(VERSION 3.25)

# include("cmake/HunterGate.cmake")
#
# HunterGate(
#     URL "https://github.com/cpp-pm/hunter/archive/v0.23.297.tar.gz"
#     SHA1 "3319fe6a3b08090df7df98dee75134d68e2ef5a3"
# )


set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

add_subdirectory(exts/corrade EXCLUDE_FROM_ALL)

set(MAGNUM_WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_GLTFIMPORTER ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_ANYSCENEIMPORTER ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_ANYIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_VK ON CACHE BOOL "" FORCE)
set(MAGNUM_BUILD_PLUGINS_STATIC ON CACHE BOOL "" FORCE)

add_subdirectory(exts/magnum EXCLUDE_FROM_ALL)

set(MAGNUM_WITH_GLTFIMPORTER ON CACHE BOOL "" FORCE)

add_subdirectory(exts/magnum-plugins EXCLUDE_FROM_ALL)

project(animate)

find_package(Magnum REQUIRED
    GL
    MeshTools
    Primitives
    Shaders
    Trade
    AnySceneImporter
    AnyImageImporter
    SceneTools
    SceneGraph
    Text
    Sdl2Application)

find_package(MagnumPlugins REQUIRED
    GltfImporter
    )

add_executable(animate src/main.cpp)
target_link_libraries(animate PRIVATE
    Magnum::Application
    Magnum::GL
    Magnum::Trade
    Magnum::AnyImageImporter
    Magnum::AnySceneImporter
    Magnum::MeshTools
    Magnum::Shaders
    Magnum::Magnum
    MagnumPlugins::GltfImporter)

# add_dependencies(animate MagnumPlugins::GltfImporter)
