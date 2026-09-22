enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;

fn Pack4x16float(v : vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}

fn Unpack4x16float(v : vec2<u32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
}
struct dst_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(0) var<storage, read_write> dst_tensor_buffer : dst_tensor_buffer_vector;
struct src_tensor_1_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(1) var<storage, read> src_tensor_1_buffer : src_tensor_1_buffer_vector;
struct src_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(2) var<storage, read> src_tensor_buffer : src_tensor_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.z || Y >= U.i0.x || S >= U.i0.y) { 
    return; 
  } 
  var src : vec4<f16>= src_tensor_buffer.data[(((S) * U.i1.y + (Y)) * U.i1.z + (X))];
  {

   var src_final : vec4<f16>;
  {  
  {var second_value : vec4<f16>= src_tensor_1_buffer.data[((((S)) * U.i0.w + ((Y))) * U.i1.x + ((X)))];
  src_final = src - second_value;}
  }
  dst_tensor_buffer.data[(((S) * U.i0.x + (Y)) * U.i0.z + (X))] = src_final;
};
} 
