enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_1_link3_image2d : texture_2d<f32>;
@group(0) @binding(2) var src_tensor_image2d : texture_2d<f32>;
struct Scalars {
  f0 : vec4<f32>,
  f1 : vec4<f32>,
  f2 : vec4<f32>,
  i3 : vec4<i32>,
  i4 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i3.z || Y >= U.i3.x || S >= U.i3.y) { 
    return; 
  } 
  var src : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i4.x + (S))), 0));
  {

   var src_final : vec4<f16>;
  {  
  
   var interm_value_link1 : vec4<f16>;
  {
  var clamped_value : vec4<f16>= min(vec4<f16>(f16(U.f1.z), f16(U.f1.z), f16(U.f1.z), f16(U.f1.z)), max(vec4<f16>(f16(U.f2.x), f16(U.f2.x), f16(U.f2.x), f16(U.f2.x)), src));
  var quantized_value : vec4<f16>= round((clamped_value - vec4<f16>(f16(U.f2.x), f16(U.f2.x), f16(U.f2.x), f16(U.f2.x))) * vec4<f16>(f16(U.f1.x), f16(U.f1.x), f16(U.f1.x), f16(U.f1.x)));
  
  var dequantized_value : vec4<f16>= quantized_value * vec4<f16>(f16(U.f2.z), f16(U.f2.z), f16(U.f2.z), f16(U.f2.z)) + vec4<f16>(f16(U.f2.x), f16(U.f2.x), f16(U.f2.x), f16(U.f2.x));
  interm_value_link1 = dequantized_value;}
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = max(interm_value_link1, second_val);}
  
   var interm_value_link3 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z));
  interm_value_link3 = min(interm_value_link2, second_val);}
  
   var interm_value_link6 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link3_image2d, vec2<i32>(((X)), (((Y)) * U.i3.w + ((S)))), 0));
  interm_value_link6 = interm_value_link3 * second_value;}
  
   var interm_value_link6_link1 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.w), f16(U.f0.w), f16(U.f0.w), f16(U.f0.w));
  interm_value_link6_link1 = max(interm_value_link6, second_val);}
  
   var interm_value_link6_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  interm_value_link6_link2 = min(interm_value_link6_link1, second_val);}
  {
  var clamped_value : vec4<f16>= min(vec4<f16>(f16(U.f1.w), f16(U.f1.w), f16(U.f1.w), f16(U.f1.w)), max(vec4<f16>(f16(U.f2.y), f16(U.f2.y), f16(U.f2.y), f16(U.f2.y)), interm_value_link6_link2));
  var quantized_value : vec4<f16>= round((clamped_value - vec4<f16>(f16(U.f2.y), f16(U.f2.y), f16(U.f2.y), f16(U.f2.y))) * vec4<f16>(f16(U.f1.y), f16(U.f1.y), f16(U.f1.y), f16(U.f1.y)));
  
  var dequantized_value : vec4<f16>= quantized_value * vec4<f16>(f16(U.f2.w), f16(U.f2.w), f16(U.f2.w), f16(U.f2.w)) + vec4<f16>(f16(U.f2.y), f16(U.f2.y), f16(U.f2.y), f16(U.f2.y));
  src_final = dequantized_value;}
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i3.y + (S))), vec4<f32>(src_final));
};
} 
