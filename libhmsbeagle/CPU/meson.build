cpu = shared_library('hmsbeagle-cpu',
                    'BeagleCPUPlugin.cpp',
                    include_directories: include_directories('.'),
                    dependencies: [eigen,hmsbeagle],
                    soversion: plugin_version_extended,
#                    name_suffix: plugin_suffix,
                    install_dir: install_dir,
                    install: true)

cpu_sse = shared_library('hmsbeagle-cpu-sse',
                        'BeagleCPUSSEPlugin.cpp',
                        include_directories: include_directories('.'),
                        dependencies: [eigen, hmsbeagle],
                        soversion: plugin_version_extended,
#                        name_suffix: plugin_suffix,
                        install_dir: install_dir,
                        install: true)


if get_option('build-action')

  cpu_sse = shared_library('hmsbeagle-cpu-action',
                           'BeagleCPUActionPlugin.cpp',
                           include_directories: include_directories('.'),
                           dependencies: [eigen, hmsbeagle],
                           soversion: plugin_version_extended,
#                           name_suffix: plugin_suffix,
                           install_dir: install_dir,
                           install: true)

endif
