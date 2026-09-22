enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct dst_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(0) var<storage, read_write> dst_buffer : dst_buffer_vector;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> params_buffer : params_buffer_vector;
struct src_buffer_vector {
  data: array<vec4<i32>>,
};
@group(0) @binding(2) var<storage, read> src_buffer : src_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var src_token_index : i32= i32(reserved_gid.x);
  var batch_index : i32= i32(reserved_gid.y);
  var src_batch : i32= batch_index % 1;
  ;
  if (src_token_index >= U.i0.z || batch_index >= U.i0.w) {
    return;
  }

  var token : i32= src_buffer.data[(((0) * U.i0.y + (0)) * U.i0.z + (src_token_index))].x;
  var dst_token_index : i32= src_token_index + params_buffer.data[0];
  if (dst_token_index < U.i0.x) {
    dst_buffer.data[dst_token_index * U.i0.w + batch_index] = token;
  }
}