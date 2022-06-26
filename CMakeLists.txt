cmake_minimum_required(VERSION 3.20)
project(CPPractice)

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(/usr/local/Cellar/jsoncpp/1.9.4_1/include )
LINK_DIRECTORIES(  /usr/local/Cellar/jsoncpp/1.9.4_1/lib )

set(MY_SOURCES
        ${MY_SOURCES}
        out_put.h
        out_put.cpp my_algrithm/sort/heap_sorter.h)

add_subdirectory(include)
add_subdirectory(company)
add_subdirectory(model)
add_subdirectory(stlib)
add_subdirectory(templ)
add_subdirectory(foo)
add_subdirectory(my_algrithm)

add_library(libCPPractice
        OBJECT
        ${MY_SOURCES})

file(GLOB LIBRARIES "/usr/local/Cellar/jsoncpp/1.9.4_1/lib/*.dylib")
message("LIBRARIES = ${LIBRARIES}")

add_executable(CPPractice main.cpp $<TARGET_OBJECTS:libCPPractice>)

TARGET_LINK_LIBRARIES(CPPractice ${LIBRARIES})

target_include_directories(CPPractice PRIVATE)
target_include_directories(libCPPractice PRIVATE)
