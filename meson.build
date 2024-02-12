project('libhmsbeagle', ['cpp', 'c', 'java'],
        version: '4.0.1',
        default_options: [
          'cpp_std=c++17',
          'warning_level=3',
          'debug=true',
          'buildtype=release',
          'b_ndebug=if-release'
        ],
        meson_version: '>= 1.1')

version_array = meson.project_version().split('.')
plugin_version = '@0@@1@'.format(version_array[0], version_array[1])

cpp = meson.get_compiler('cpp')
cc = meson.get_compiler('c')

package_name = 'libhmsbeagle'


if get_option('benchmark')
  add_project_arguments('-DBEAGLE_BENCHMARK', language: 'cpp')
endif

if get_option('debug-flow')
  add_project_arguments('-DBEAGLE_DEBUG_FLOW', language: 'cpp')
endif

if get_option('debug-values')
  add_project_arguments('-DBEAGLE_DEBUG_VALUES', language: 'cpp')
endif

                      
add_project_arguments(['-DPACKAGE_BUGREPORT="beagle-dev@googlegroups.com"',
                       '-DPACKAGE_NAME="@0@"'.format(meson.project_name()),
                       '-DPACKAGE_STRING="@0@ @1@"'.format(meson.project_name(), meson.project_version()),
                       '-DPACKAGE_TARNAME="@0@"'.format(meson.project_name()),
                       '-DPACKAGE_VERSION="@0@"'.format(meson.project_version()),
                       '-DPACKAGE_URL=""',
                       '-DPLUGIN_VERSION="@0@"'.format(plugin_version)],
                       language: 'cpp')

#set(CMAKE_BUILD_TYPE RelWithDebInfo)

if host_machine.system() == 'darwin'
  add_project_arguments('-DDLS_MACOS', language: 'cpp')
endif

jni = dependency('jni', required: get_option('build-jni'))
#    message (STATUS "JAVA_HOME=$ENV{JAVA_HOME}")
#    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
#    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")

openmp = dependency('openmp', required: get_option('build-openmp'))

if get_option('build-action')
  eigen = dependency('eigen3')
  add_project_arguments('-DBEAGLE_ACTION', language: 'cpp')
endif


# link_dirs = files('libhmsbeagle', 'libhmsbeagle/CPU', 'libhmsbeagle/GPU', 'libhmsbeagle/JNI')


# This statement let cmake look for libraries in the provided directories first.
# if failed, RUNTIME_LIB = NO_DEFAULT_PATH
#FIND_LIBRARY(RUNTIME_LIB rt /usr/lib  /usr/local/lib NO_DEFAULT_PATH)

if host_machine.system() == 'darwin'

    # Check if running under rosetta on arm64 hardware
  if host_machine.cpu_family().startswith('arm') or host_machine.cpu_family().startswith('aarch')
    #set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    message('macOS universal (x86_64 / arm64) build')
  else
    message('macOS x86_64 build')
  endif

  # Seems necessary from macOS 10.15
  # SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  extra_rpath = '@loaderpath/../lib/'
endif

if get_option('optimize-for-native-arch')
  add_project_arguments(cpp.get_supported_arguments(['-march=native','/arch:AVX']), language: 'cpp')
endif

add_project_arguments(cpp.get_supported_arguments(['-Wno-unused-parameter',
                                                   '-Wno-unknown-pragmas',
                                                   '-Wno-sign-compare']), language: 'cpp')

subdir('libhmsbeagle')

#?? link_libraries(hmsbeagle)

#?? enable_testing()

subdir('examples')

#set(CPACK_PACKAGE_VERSION "${BEAGLE_VERSION}")
#set(CPACK_PACKAGE_MAJOR "${BEAGLE_VERSION_MAJOR}")
#set(CPACK_PACKAGE_MINOR "${BEAGLE_VERSION_MINOR}")
#set(CPACK_PACKAGE_PATCH "${BEAGLE_VERSION_PATCH}")

#if(WIN32)
#	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
#	set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

#    set(CPACK_GENERATOR WIX)
#    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Common Files/libhmsbeagle")
#    set(CPACK_WIX_UPGRADE_GUID "57FC5933-B420-4F10-97F3-B5DDCFF2AE13")
#    set(CPACK_WIX_LICENSE_RTF "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/LICENSE.rtf")
#    set(CPACK_WIX_UI_BANNER "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/banner.jpg")
#    set(CPACK_WIX_UI_DIALOG "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")
#    set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/patch.xml")
#endif(WIN32)

#if(APPLE)
#	set(CPACK_GENERATOR productbuild)
#	set(CPACK_BUNDLE_NAME "${BEAGLE_PACKAGE_NAME}")
#	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/LICENSE.rtf")
#	set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/project/beagle-macos/README.txt")
#
#	set(CPACK_PRODUCTBUILD_BACKGROUND_UTI "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")
#
#	get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
#
#	include(CPackComponent)
#
#	cpack_add_component(main
#			DISPLAY_NAME "Main library"
#			DESCRIPTION "BEAGLE main library"
#			REQUIRED)
#
#	cpack_add_component(cpu
# 			DISPLAY_NAME "CPU plugin"
#		  	DESCRIPTION "CPU plugin"
#  			REQUIRED)
#
#  	cpack_add_component(cpu_sse
#  			DISPLAY_NAME "CPU-SSE plugin"
#  			DESCRIPTION "CPU-SSE plugin")
#
#	cpack_add_component(cpu_action
#			DISPLAY_NAME "CPU-Action plugin"
#			DESCRIPTION "CPU-Action plugin")
#
#  	cpack_add_component(cuda
#  			DISPLAY_NAME "CUDA plugin"
#  			DESCRIPTION "CUDA plugin")
#
#  	cpack_add_component(opencl
#  			DISPLAY_NAME "OpenCL plugin"
#  			DESCRIPTION "OpenCL plugin")
#
#	set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
#endif(APPLE)
#
#include(CPack)
