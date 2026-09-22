enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var r_s_x : i32;var r_s_y : i32;var r_s_s : i32;var r_s_b : i32;

  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.z || Y >= U.i0.x || S >= U.i0.y) { 
    return; 
  }{
  var xc_link1 : i32;var yc_link1 : i32;var sc_link1 : i32;
  {
  sc_link1 = (S);
  xc_link1 = (Y);
  yc_link1 = (X);
  }
  var out_linear : i32= 0;

  out_linear = ((out_linear * 64 + yc_link1) * 35 + xc_link1) * 64 + sc_link1;
  r_s_s = out_linear % U.i1.x;
  out_linear = out_linear / U.i1.x;
  r_s_x = out_linear % U.i1.y;
  out_linear = out_linear / U.i1.y;
  r_s_y = out_linear % U.i0.w;

} 
  var src : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((r_s_x), ((r_s_y) * U.i1.x + (r_s_s))), 0));
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.y + (S))), vec4<f32>(src));
} 
