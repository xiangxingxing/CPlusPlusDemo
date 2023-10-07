add_subdirectory(link_list)
add_subdirectory(tree)
add_subdirectory(sort)

set(MY_SOURCES
        ${MY_SOURCES}
        my_algrithm/lru_cache.cpp
        my_algrithm/lru_cache.h
        my_algrithm/my_stack.h
        my_algrithm/my_stack.cpp
        my_algrithm/lint_code_manager.h
        my_algrithm/lint_code_manager.cpp
        my_algrithm/singleton.h
        PARENT_SCOPE
        )