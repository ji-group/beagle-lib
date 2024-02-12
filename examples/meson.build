hmctest = executable('hmctest',
                     'hmctest/hmctest.cpp',
                     dependencies: hmsbeagle)

hmctest5 = executable('hmctest5',
	              'hmctest/hmctest5.cpp',
                      dependencies: hmsbeagle)

actiontest = executable('actiontest',
	                'actiontest/actiontest.cpp',
                        dependencies: [eigen,hmsbeagle],
                        link_with: cpu)

#add_executable(hmcGaptest
#    	hmctest/hmcGapTest.cpp)

#add_executable(tinytest
#        tinytest/tinytest.cpp)

#add_executable(hmctestMedium
#        hmctest/hmctestMedium.cpp)

#add_executable(genomictest
#        genomictest/genomictest.cpp
#        genomictest/linalg.cpp
#        genomictest/linalg.h
#        )

executable('synthetictest',
	   ['synthetictest/synthetictest.cpp',
            'synthetictest/linalg.cpp'],
           dependencies: hmsbeagle
           )

#add_executable(complextest
#        complextest/complextest.cpp)

test('hmctest', hmctest)

#target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(hmcGaptest hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)
