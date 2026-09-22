enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
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
  var src : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.w + (S))), 0));
  {

   var src_final : vec4<f16>;
  {  
  {
  var clamped_value : vec4<f16>= min(vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y)), max(vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z)), src));
  var quantized_value : vec4<f16>= round((clamped_value - vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z))) * vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x)));
  
  var dequantized_value : vec4<f16>= quantized_value * vec4<f16>(f16(U.f0.w), f16(U.f0.w), f16(U.f0.w), f16(U.f0.w)) + vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z));
  src_final = dequantized_value;}
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.y + (S))), vec4<f32>(src_final));
};
} 
