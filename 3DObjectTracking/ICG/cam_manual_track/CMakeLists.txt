cmake_minimum_required(VERSION 3.5)

add_executable(run_on_recorded_sequence run_on_recorded_sequence.cpp)
target_link_libraries(run_on_recorded_sequence PUBLIC icg)

add_executable(run_generated_tracker run_generated_tracker.cpp)
target_link_libraries(run_generated_tracker PUBLIC icg)

#if (USE_AZURE_KINECT)
if (USE_REALSENSE)
    add_executable(record_camera_sequence record_camera_sequence.cpp)
    target_link_libraries(record_camera_sequence PUBLIC icg)

    add_executable(run_on_camera_sequence run_on_camera_sequence.cpp)
    target_link_libraries(run_on_camera_sequence PUBLIC icg)
endif ()
