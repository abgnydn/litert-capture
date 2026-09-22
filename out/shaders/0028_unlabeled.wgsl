enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_image2d : texture_storage_2d<rgba32sint, write>;
struct src_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> src_buffer : src_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
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
  var res_value : vec4<i32>;
  var offset : i32= src_buffer.data[U.i0.w];
  res_value.x = i32(offset + X);
  res_value.y = i32(offset + X);
  res_value.z = i32(offset + X);
  res_value.w = i32(offset + X);
  textureStore(dst_image2d, vec2<i32>((X), ((Y) * U.i0.y + (S))), res_value);
}
