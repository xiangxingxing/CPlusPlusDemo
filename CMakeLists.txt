cmake_minimum_required(VERSION 3.20)
project(untitled)

set(CMAKE_CXX_STANDARD 14)

add_executable(untitled main.cpp out_put.cpp out_put.h foo/Student.cpp foo/Student.h templ/Point.cpp templ/Point.h templ/List.cpp templ/List.h company/worker_manager.h company/worker_manager.cpp company/worker.h company/employee.h company/employee.cpp company/manager.h company/manager.cpp company/boss.cpp company/boss.h stlib/std_manager.h stlib/std_manager.cpp stlib/example/player.h my_algrithm/tree/tree_node.h my_algrithm/tree/tree_coder.cpp my_algrithm/lru_cache.h my_algrithm/lru_cache.cpp)
