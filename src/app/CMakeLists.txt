add_library(lmp_app_host STATIC host_helpers.cpp)
target_include_directories(lmp_app_host PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_app_host PRIVATE lmp_warnings lmp_assertions PUBLIC lmp_platform)

add_executable(Piper MACOSX_BUNDLE
  main.mm
  PiperWindow.mm
  SidecarBridge.mm
)
target_include_directories(Piper PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_options(Piper PRIVATE -Wall -Wextra -Werror -fobjc-arc)
target_link_libraries(Piper PRIVATE
  lmp_app_host
  "-framework Cocoa"
  "-framework WebKit"
  "-framework UniformTypeIdentifiers"
)
set_target_properties(Piper PROPERTIES
  MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
  XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

add_dependencies(Piper lmp_sidecar)

# Bundle lmp_sidecar + mlx.metallib right beside the app executable so Piper.app runs
# standalone with one in-process MLX model child and zero external binaries needed.
add_custom_command(TARGET Piper POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:lmp_sidecar>" "$<TARGET_FILE_DIR:Piper>/lmp_sidecar"
  COMMENT "Bundling lmp_sidecar into Piper.app"
  VERBATIM)

if(TARGET mlx)
  add_custom_command(TARGET Piper POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LMP_MLX_METALLIB}" "$<TARGET_FILE_DIR:Piper>/mlx.metallib"
    COMMENT "Bundling mlx.metallib into Piper.app"
    VERBATIM)
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/app/dist/index.html")
  add_custom_command(TARGET Piper POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "$<TARGET_BUNDLE_CONTENT_DIR:Piper>/Resources/ui"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${PROJECT_SOURCE_DIR}/app/dist" "$<TARGET_BUNDLE_CONTENT_DIR:Piper>/Resources/ui"
    COMMENT "Staging app/dist UI into Piper.app/Contents/Resources/ui"
    VERBATIM)
endif()

find_program(LMP_NPM npm)
if(LMP_NPM)
  add_custom_target(piper-ui
    COMMAND "${LMP_NPM}" --prefix "${PROJECT_SOURCE_DIR}/app" run build
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/app"
    COMMENT "Building Piper Desktop web UI (app/dist)"
    VERBATIM)

  add_custom_target(piper-app
    DEPENDS piper-ui Piper
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "$<TARGET_BUNDLE_CONTENT_DIR:Piper>/Resources/ui"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${PROJECT_SOURCE_DIR}/app/dist" "$<TARGET_BUNDLE_CONTENT_DIR:Piper>/Resources/ui"
    COMMENT "Bundling built UI into Piper.app"
    VERBATIM)
endif()

add_custom_target(install-piper-app
  DEPENDS Piper
  COMMAND "${CMAKE_COMMAND}" -E remove_directory "/Applications/Piper.app"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "$<TARGET_BUNDLE_DIR:Piper>" "/Applications/Piper.app"
  COMMENT "Installing Piper.app into /Applications/Piper.app"
  VERBATIM)
