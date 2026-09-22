enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_1_link2_image2d : texture_2d<f32>;
struct scale_buffer_vector {
  data: array<vec4<f32>>,
};
@group(0) @binding(2) var<storage, read> scale_buffer : scale_buffer_vector;
struct weights_buffer_vector {
  data: array<vec2<u32>>,
};
@group(0) @binding(3) var<storage, read> weights_buffer : weights_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var DST_X : i32= i32(reserved_gid.x);
  var DST_S : i32= i32(reserved_gid.y);
  var DST_Y : i32= i32(reserved_gid.z);
  if (DST_X >= U.i1.y || DST_S >= U.i1.x / 4 || DST_Y != 0) {return;}
  var SRC_X : i32= DST_X;
  var w0 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var w1 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var w2 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var w3 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var weight_scale : f16;
    {
  var slice_coord_TMP : i32= (SRC_X) / 4;
  var sub_ch_coord_TMP : i32= (SRC_X) % 4;
  weight_scale = vec4<f16>(scale_buffer.data[(slice_coord_TMP)])[sub_ch_coord_TMP];
  };

  var w : vec2<u32>= weights_buffer.data[SRC_X * U.i1.x / 4 + DST_S];
  w = w ^ vec2<u32>(0x88888888, 0x88888888);
    {

  var wt0 : vec4<f16>;
  var wt1 : vec4<f16>;
  wt0.x = f16((w.x) & 255u) * f16(0.0625);
  wt0.y = f16((w.x >>  8u) & 255u) * f16(0.0625);
  wt0.z = f16((w.x >> 16u) & 255u) * f16(0.0625);
  wt0.w = f16((w.x >> 24u) & 255u) * f16(0.0625);
  wt1.x = f16((w.y) & 255u) * f16(0.0625);
  wt1.y = f16((w.y >>  8u) & 255u) * f16(0.0625);
  wt1.z = f16((w.y >> 16u) & 255u) * f16(0.0625);
  wt1.w = f16((w.y >> 24u) & 255u) * f16(0.0625);

  w0.y = floor(wt0.x);
  w0.w = floor(wt0.y);
  w0.x = (wt0.x - w0.y) * f16(16.0);
  w0.z = (wt0.y - w0.w) * f16(16.0);
  w1.y = floor(wt0.z);
  w1.w = floor(wt0.w);
  w1.x = (wt0.z - w1.y) * f16(16.0);
  w1.z = (wt0.w - w1.w) * f16(16.0);
  w2.y = floor(wt1.x);
  w2.w = floor(wt1.y);
  w2.x = (wt1.x - w2.y) * f16(16.0);
  w2.z = (wt1.y - w2.w) * f16(16.0);
  w3.y = floor(wt1.z);
  w3.w = floor(wt1.w);
  w3.x = (wt1.z - w3.y) * f16(16.0);
  w3.z = (wt1.w - w3.w) * f16(16.0);
  }
;
  var bias : f16= -weight_scale * 8.0;
  w0 = w0 * weight_scale + bias;
  w1 = w1 * weight_scale + bias;
  w2 = w2 * weight_scale + bias;
  w3 = w3 * weight_scale + bias;
  {

   var w0_final : vec4<f16>;
  {  
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = w0 * second_val;}
  
   var interm_value_link3 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link2_image2d, vec2<i32>(((DST_X)), (((0)) * U.i1.z + ((DST_S * 4 + 0)))), 0));
  interm_value_link3 = interm_value_link2 + second_value;}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  w0_final = interm_value_link3 * second_val;}
  }
  textureStore(dst_image2d, vec2<i32>((DST_X), ((0) * U.i1.x + (DST_S * 4 + 0))), vec4<f32>(w0_final));
};
  {

   var w1_final : vec4<f16>;
  {  
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = w1 * second_val;}
  
   var interm_value_link3 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link2_image2d, vec2<i32>(((DST_X)), (((0)) * U.i1.z + ((DST_S * 4 + 1)))), 0));
  interm_value_link3 = interm_value_link2 + second_value;}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  w1_final = interm_value_link3 * second_val;}
  }
  textureStore(dst_image2d, vec2<i32>((DST_X), ((0) * U.i1.x + (DST_S * 4 + 1))), vec4<f32>(w1_final));
};
  {

   var w2_final : vec4<f16>;
  {  
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = w2 * second_val;}
  
   var interm_value_link3 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link2_image2d, vec2<i32>(((DST_X)), (((0)) * U.i1.z + ((DST_S * 4 + 2)))), 0));
  interm_value_link3 = interm_value_link2 + second_value;}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  w2_final = interm_value_link3 * second_val;}
  }
  textureStore(dst_image2d, vec2<i32>((DST_X), ((0) * U.i1.x + (DST_S * 4 + 2))), vec4<f32>(w2_final));
};
  {

   var w3_final : vec4<f16>;
  {  
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  interm_value_link2 = w3 * second_val;}
  
   var interm_value_link3 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link2_image2d, vec2<i32>(((DST_X)), (((0)) * U.i1.z + ((DST_S * 4 + 3)))), 0));
  interm_value_link3 = interm_value_link2 + second_value;}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  w3_final = interm_value_link3 * second_val;}
  }
  textureStore(dst_image2d, vec2<i32>((DST_X), ((0) * U.i1.x + (DST_S * 4 + 3))), vec4<f32>(w3_final));
};
}