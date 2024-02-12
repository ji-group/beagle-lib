
plugin_version_extended = plugin_version

if host_machine.system() == 'windows'
  add_project_arguments(['-D_EXPORTING', '-D_CONSOLE'], language: 'cpp')

  plugin_suffix='64-@0@.dll'.format(plugin_version)
  install_dir='.'
  header_dir='.'

  ltdl = dependency('', required: false)
  plugin_source = files(['plugin/WinSharedLibrary.cpp'])

  if get_option('build-jni')
    integrated_jni_source=files(['JNI/beagle_BeagleJNIWrapper.cpp'])
  else
    integrated_jni_source=files()
  endif
else
  plugin_suffix='so'
  install_dir='lib'
  header_dir='include/libhmsbeagle-1/libhmsbeagle'

  plugin_source = files(['plugin/UnixSharedLibrary.cpp'])
  if get_option('build-jni')
    integrated_jni_source=[]
  else
    integrated_jni_source=files()
  endif

  if get_option('experimental-ltdl')
    ltdl = dependency('ltdl')
  else
    ltdl = dependency('', required: false)
  endif

  if ltdl.found()
    add_project_arguments(['-DHAVE_LIBLTDL'], language: 'cpp')
    message('Using libtools for plugins: @0@'.format(ltdl))
  else
    message('Not using libtools for plugins')
  endif

  if host_machine.system() != 'darwin'
    plugin_version_extended = plugin_version + '.0.0'
  endif

endif

# if not windows and not apple
dl = dependency('dl')

# build minimum functional beagle-lib
beagle_src = files(['beagle.cpp',
                    'plugin/Plugin.cpp',
                    'benchmark/BeagleBenchmark.cpp',
                    'benchmark/linalg.cpp'
                   ]) + integrated_jni_source + plugin_source

beagle_inc = include_directories(['..', '.','plugin','benchmark'])

install_rpath = ''
build_rpath=''
if host_machine.system() == 'linux'
  build_rpath = '$ORIGIN:$ORIGIN/CPU:$ORIGIN/GPU:$ORIGIN/JNI'
  install_rpath = '$ORIGIN'
endif


libhmsbeagle = shared_library('hmsbeagle',
                              beagle_src,
                              include_directories: beagle_inc,
                              dependencies: [jni, ltdl, dl, eigen],
                              install_dir: install_dir,
#                              name_suffix: plugin_suffix,
                              build_rpath: build_rpath,
                              install_rpath: install_rpath,
                              install: true)


install_headers(files(['beagle.h','platform.h']), subdir: 'hmsbeagle-1')

hmsbeagle = declare_dependency(include_directories: include_directories('..'),
                               link_with: libhmsbeagle)
                           


#add_library(hmsbeagle SHARED
#        beagle.cpp
#        beagle.h
#        BeagleImpl.h
#        platform.h

#        benchmark/BeagleBenchmark.h
#        benchmark/BeagleBenchmark.cpp
#        benchmark/linalg.h
#        benchmark/linalg.cpp
#        benchmark/BenchmarkHelper.h

#        ${BEAGLE_INTEGRATED_JNI_SOURCE}

#        plugin/BeaglePlugin.h
#        plugin/Plugin.cpp
#        plugin/Plugin.h
#        plugin/SharedLibrary.h
#        ${BEAGLE_PLUGIN_SOURCE}
#        )


if host_machine.system() == 'windows'
  plugin_suffix='64.dll'
endif

subdir('CPU')
#subdir('GPU')
#if get_option('build-jni')
#  subdir('JNI')
#endif


pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : libhmsbeagle,
                 version : '1.0',
                 name : 'hmsbeagle',
                 filebase : 'hmsbeagle',
                 description : 'A library to calculate phylogenetic likelihoods.')
