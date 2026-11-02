cmake_minimum_required(VERSION 3.5)
project(huibo_description)
find_package(ament_cmake REQUIRED)

# Install launch files.
install(DIRECTORY launch  meshes urdf rviz config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
