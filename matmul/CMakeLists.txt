set(MATMUL_APP_NAME "matmul_bench")
set(SPDM_APP_NAME "spdm_bench")

#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -fno-omit-frame-pointer -static-liblsan -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -fno-omit-frame-pointer -static-liblsan -fsanitize=address")

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -mavx512f -fno-omit-frame-pointer")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")

file(GLOB MATMUL_BENCH_SRC
     ${CMAKE_CURRENT_SOURCE_DIR}/matmul_bench.cpp
     )

add_executable(${MATMUL_APP_NAME} ${MATMUL_BENCH_SRC})
target_link_libraries(${MATMUL_APP_NAME} dnnl_brgemm ${MKL_LIBRARIES})

file(GLOB SPDM_BENCH_SRC
     ${CMAKE_CURRENT_SOURCE_DIR}/spdm_bench.cpp
     )

add_executable(${SPDM_APP_NAME} ${SPDM_BENCH_SRC})
target_link_libraries(${SPDM_APP_NAME} dnnl_brgemm ${MKL_LIBRARIES})