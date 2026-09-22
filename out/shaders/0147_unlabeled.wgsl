enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba32float, write>;
struct src_tensor_buffer_vector {
  data: array<vec4<f32>>,
};
@group(0) @binding(1) var<storage, read> src_tensor_buffer : src_tensor_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.z || Y >= U.i0.x || S >= U.i0.y) { 
    return; 
  } 
  var s_x : i32= X * U.i2.x + U.i0.w;
  var s_y : i32= Y * U.i2.y + U.i1.x;
  var s_z : i32= S + U.i1.y;
  var result : vec4<f32>= src_tensor_buffer.data[(((s_z) * U.i1.z + (s_y)) * U.i1.w + (s_x))];
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.y + (S))), result);
}
