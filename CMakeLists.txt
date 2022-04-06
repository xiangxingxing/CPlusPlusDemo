cmake_minimum_required(VERSION 3.20)
project(CPPractice)

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(/usr/local/Cellar/jsoncpp/1.9.4_1/include )
LINK_DIRECTORIES(  /usr/local/Cellar/jsoncpp/1.9.4_1/lib )

add_subdirectory(include)
add_subdirectory(company)
add_subdirectory(model)
add_subdirectory(stlib)
add_subdirectory(templ)
add_subdirectory(foo)
add_subdirectory(my_algrithm)

file(GLOB LIBRARIES "/usr/local/Cellar/jsoncpp/1.9.4_1/lib/*.dylib")
message("LIBRARIES = ${LIBRARIES}")

add_executable(CPPractice main.cpp out_put.cpp out_put.h ${MY_SOURCES})

TARGET_LINK_LIBRARIES(CPPractice ${LIBRARIES})
