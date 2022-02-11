add_subdirectory(example)

set(MY_SOURCES
        ${MY_SOURCES}
        stlib/std_manager.h
        stlib/std_manager.cpp
        PARENT_SCOPE
        )