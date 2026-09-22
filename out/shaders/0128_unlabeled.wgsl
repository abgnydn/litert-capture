enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
struct Scalars {
  f0 : vec4<f32>,
  f1 : vec4<f32>,
  i2 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i2.z || Y >= U.i2.x || S >= U.i2.y) { 
    return; 
  } 
  var src : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i2.w + (S))), 0));
  {

   var src_final : vec4<f16>;
  {  
  
   var interm_value_link1 : vec4<f16>;
  {
  var clamped_value : vec4<f16>= min(vec4<f16>(f16(U.f0.w), f16(U.f0.w), f16(U.f0.w), f16(U.f0.w)), max(vec4<f16>(f16(U.f1.x), f16(U.f1.x), f16(U.f1.x), f16(U.f1.x)), src));
  var quantized_value : vec4<f16>= round((clamped_value - vec4<f16>(f16(U.f1.x), f16(U.f1.x), f16(U.f1.x), f16(U.f1.x))) * vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z)));
  
  var dequantized_value : vec4<f16>= quantized_value * vec4<f16>(f16(U.f1.y), f16(U.f1.y), f16(U.f1.y), f16(U.f1.y)) + vec4<f16>(f16(U.f1.x), f16(U.f1.x), f16(U.f1.x), f16(U.f1.x));
  interm_value_link1 = dequantized_value;}
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = max(interm_value_link1, second_val);}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  src_final = min(interm_value_link2, second_val);}
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i2.y + (S))), vec4<f32>(src_final));
};
} 
