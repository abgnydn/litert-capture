enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct dst_buffer_vector {
  data: array<vec4<f32>>,
};
@group(0) @binding(0) var<storage, read_write> dst_buffer : dst_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
};
@group(0) @binding(1) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X != 0 || Y != 0 || S != 0) {return;}
  var result : vec4<f32>;
  result.x = U.f0.x;
  result.y = U.f0.y;
  result.z = U.f0.z;
  result.w = U.f0.z;
  dst_buffer.data[(((0) * U.i1.x + (0)) * U.i1.y + (0))] = result;
}
