# generated from
# rosidl_cmake/cmake/template/rosidl_cmake_export_typesupport_targets.cmake.in

set(_exported_typesupport_targets
  "__rosidl_typesupport_introspection_c:person_following_robot__rosidl_typesupport_introspection_c;__rosidl_typesupport_introspection_cpp:person_following_robot__rosidl_typesupport_introspection_cpp")

# populate person_following_robot_TARGETS_<suffix>
if(NOT _exported_typesupport_targets STREQUAL "")
  # loop over typesupport targets
  foreach(_tuple ${_exported_typesupport_targets})
    string(REPLACE ":" ";" _tuple "${_tuple}")
    list(GET _tuple 0 _suffix)
    list(GET _tuple 1 _target)

    set(_target "person_following_robot::${_target}")
    if(NOT TARGET "${_target}")
      # the exported target must exist
      message(WARNING "Package 'person_following_robot' exports the typesupport target '${_target}' which doesn't exist")
    else()
      list(APPEND person_following_robot_TARGETS${_suffix} "${_target}")
    endif()
  endforeach()
endif()
