enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct output_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(0) var<storage, read_write> output_buffer : output_buffer_vector;
struct params_i32_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> params_i32_buffer : params_i32_buffer_vector;
struct tokens_buffer_vector {
  data: array<vec4<i32>>,
};
@group(0) @binding(2) var<storage, read> tokens_buffer : tokens_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var B : i32= i32(reserved_gid.x);
  var W : i32= i32(reserved_gid.y);
  ;

  if (W >= U.i0.z || B >= 1 || i32(reserved_gid.z) != 0) {return;}

  var token : vec4<i32>= tokens_buffer.data[(((0) * U.i0.y + (0)) * U.i0.z + (W))];
  var time_step : i32= params_i32_buffer.data[0] + W + 1;
  if (time_step < U.i0.x) {
    output_buffer.data[time_step * 1 + B] = token.x;
  }
}