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
  
   var interm_value_link1 : vec4<f16>;
  {
  var clamped_value : vec4<f16>= min(vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y)), max(vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z)), src));
  var quantized_value : vec4<f16>= round((clamped_value - vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z))) * vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x)));
  
  var dequantized_value : vec4<f16>= quantized_value * vec4<f16>(f16(U.f0.w), f16(U.f0.w), f16(U.f0.w), f16(U.f0.w)) + vec4<f16>(f16(U.f0.z), f16(U.f0.z), f16(U.f0.z), f16(U.f0.z));
  interm_value_link1 = dequantized_value;}
  {  var erf_in : vec4<f16>= interm_value_link1 * vec4<f16>(0.707106781186548, 0.707106781186548, 0.707106781186548, 0.707106781186548);
    var erf_out : vec4<f16>;
  
    { 
      var sign_val : vec4<f16>= sign(erf_in);
      var abs_val : vec4<f16>= abs(erf_in);
      var a1 : f16= f16(0.254829592);
      var a2 : f16= f16(-0.284496736);
      var a3 : f16= f16(1.421413741);
      var a4 : f16= f16(-1.453152027);
      var a5 : f16= f16(1.061405429);
      var p : f16= f16(0.3275911);
      var t : vec4<f16>= f16(1.0) / (f16(1.0) + p * abs_val);
      var y : vec4<f16>= f16(1.0) - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t*exp(-abs_val*abs_val);
      erf_out = y * sign_val;
    }
    src_final = vec4<f16>(0.5, 0.5, 0.5, 0.5) * interm_value_link1 * (vec4<f16>(1.0, 1.0, 1.0, 1.0) + erf_out);
  }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.y + (S))), vec4<f32>(src_final));
};
} 
