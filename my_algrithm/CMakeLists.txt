add_subdirectory(link_list)
add_subdirectory(tree)
add_subdirectory(sort)

set(MY_SOURCES
        ${MY_SOURCES}
        my_algrithm/lru_cache.cpp
        my_algrithm/lru_cache.h
        PARENT_SCOPE
        )