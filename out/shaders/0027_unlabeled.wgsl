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
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
struct src_tensor_buffer_vector {
  data: array<vec4<i32>>,
};
@group(0) @binding(1) var<storage, read> src_tensor_buffer : src_tensor_buffer_vector;
struct weights_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(2) var<storage, read> weights_buffer : weights_buffer_vector;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct weights_zero_point_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(4) var<storage, read> weights_zero_point_buffer : weights_zero_point_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(5) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var linear_xb : i32= i32(reserved_gid.x);
  var X : i32= linear_xb / 1;
  var B : i32= linear_xb % 1;
  ;
  ;
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i1.z || Y >= U.i1.x || S >= U.i1.y) {
    return;
  }
  var index : i32= src_tensor_buffer.data[(((0) * U.i2.x + (0)) * U.i2.y + (X))].x;

  
  index = max(index, 0);
  index = min(index, U.i1.w - 1);
  var weights_output_ch : i32= index;
  var weights_input_slice : i32= S;
  var weights_output_slice : i32= weights_output_ch / 4;
  var linear_i4o4 : i32= weights_input_slice * U.i2.z + weights_output_slice;
  var w0 : vec4<f32>;
  var w1 : vec4<f32>;
  var w2 : vec4<f32>;
  var w3 : vec4<f32>;
  var w : u32= weights_buffer.data[linear_i4o4];
  
  w0.x = f32((w) & 3u);
  w0.y = f32((w >>  2u) & 3u);
  w0.z = f32((w >>  4u) & 3u);
  w0.w = f32((w >>  6u) & 3u);
  w1.x = f32((w >>  8u) & 3u);
  w1.y = f32((w >> 10u) & 3u);
  w1.z = f32((w >> 12u) & 3u);
  w1.w = f32((w >> 14u) & 3u);
  w2.x = f32((w >> 16u) & 3u);
  w2.y = f32((w >> 18u) & 3u);
  w2.z = f32((w >> 20u) & 3u);
  w2.w = f32((w >> 22u) & 3u);
  w3.x = f32((w >> 24u) & 3u);
  w3.y = f32((w >> 26u) & 3u);
  w3.z = f32((w >> 28u) & 3u);
  w3.w = f32((w >> 30u) & 3u);
;

  var value : vec4<f32>;
  var output_sub_ch : i32= weights_output_ch % 4;
  if (output_sub_ch == 0) {
    value.x = w0.x;
    value.y = w1.x;
    value.z = w2.x;
    value.w = w3.x;
  } else if (output_sub_ch == 1) {
    value.x = w0.y;
    value.y = w1.y;
    value.z = w2.y;
    value.w = w3.y;
  } else if (output_sub_ch == 2) {
    value.x = w0.z;
    value.y = w1.z;
    value.z = w2.z;
    value.w = w3.z;
  } else if (output_sub_ch == 3) {
    value.x = w0.w;
    value.y = w1.w;
    value.z = w2.w;
    value.w = w3.w;
  }
  var scale : f32;
  var zero_point : f32= f32(0);
    {
  var slice_coord_TMP : i32= (index) / 4;
  var sub_ch_coord_TMP : i32= (index) % 4;
  scale = vec4<f32>(weights_scale_buffer.data[(slice_coord_TMP)])[sub_ch_coord_TMP];
  };
    {
  var slice_coord_TMP : i32= (index) / 4;
  var sub_ch_coord_TMP : i32= (index) % 4;
  zero_point = vec4<f32>(weights_zero_point_buffer.data[(slice_coord_TMP)])[sub_ch_coord_TMP];
  };
  var weight_bias : f32= -scale * (f32(2.0) + zero_point);
  value = value * scale + weight_bias;

  var dst_value : vec4<f16>= vec4<f16>(value);
  {

   var dst_value_final : vec4<f16>;
  {  
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  dst_value_final = dst_value * second_val;}
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.y + (S))), vec4<f32>(dst_value_final));
};
}