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
struct src_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(1) var<storage, read> src_tensor_buffer : src_tensor_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i1.z || Y >= U.i1.x || S >= U.i1.y) { 
    return; 
  } 
  var src : vec4<f16>= src_tensor_buffer.data[(((S) * U.i1.w + (Y)) * U.i2.x + (X))];
  {

   var src_final : vec4<f16>;
  {  
  
   var interm_value_link1 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link1 = src * second_val;}
  
   var interm_value_link2 : vec4<f16>;
  {var tanh_out : vec4<f32>;
  var tanh_in : vec4<f32>= vec4<f32>(interm_value_link1);
  var in_fp32 : vec4<f32>= clamp(tanh_in, vec4<f32>(-10.0, -10.0, -10.0, -10.0), vec4<f32>(10.0, 10.0, 10.0, 10.0));
  tanh_out = tanh(in_fp32);
  interm_value_link2 = vec4<f16>(tanh_out);
  }
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  src_final = interm_value_link2 * second_val;}
  }
  dst_tensor_buffer.data[(((S) * U.i1.x + (Y)) * U.i1.z + (X))] = src_final;
};
} 
