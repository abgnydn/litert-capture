enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct dst_buffer_vector {
  data: array<vec4<i32>>,
};
@group(0) @binding(0) var<storage, read_write> dst_buffer : dst_buffer_vector;
struct src_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> src_buffer : src_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var linear_xb : i32= i32(reserved_gid.x);
  var X : i32= linear_xb / 1;
  var B : i32= linear_xb % 1;
  ;
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.z || Y >= U.i0.x || S >= U.i0.y) {
    return;
  }
  var res_value : vec4<i32>;
  res_value.x = 0;
  res_value.y = 0;
  res_value.z = 0;
  res_value.w = 0;
  var token_index : i32= X + U.i0.w;
  if (token_index < U.i1.x) {
    res_value.x = src_buffer.data[token_index];
  }
  dst_buffer.data[(((S) * U.i0.x + (Y)) * U.i0.z + (X))] = res_value;
}