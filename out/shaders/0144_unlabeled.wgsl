enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
struct src_tensor_buffer_vector {
  data: array<vec4<f32>>,
};
@group(0) @binding(1) var<storage, read> src_tensor_buffer : src_tensor_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.z || Y >= U.i0.x || S >= U.i0.y) { 
    return; 
  } 
  var s_x : i32= X * U.i2.y + U.i0.w;
  var s_y : i32= Y * U.i2.z + U.i1.x;
  var result : vec4<f32>;
  {
    var s_ch : i32= min((S * 4 + 0) * U.i2.w + U.i1.y, U.i1.z - 1);
      {
  var slice_coord_TMP : i32= (s_ch) / 4;
  var sub_ch_coord_TMP : i32= (s_ch) % 4;
  result.x = src_tensor_buffer.data[(((slice_coord_TMP) * U.i1.w + (s_y)) * U.i2.x + (s_x))][sub_ch_coord_TMP];
  };
  }
  {
    var s_ch : i32= min((S * 4 + 1) * U.i2.w + U.i1.y, U.i1.z - 1);
      {
  var slice_coord_TMP : i32= (s_ch) / 4;
  var sub_ch_coord_TMP : i32= (s_ch) % 4;
  result.y = src_tensor_buffer.data[(((slice_coord_TMP) * U.i1.w + (s_y)) * U.i2.x + (s_x))][sub_ch_coord_TMP];
  };
  }
  {
    var s_ch : i32= min((S * 4 + 2) * U.i2.w + U.i1.y, U.i1.z - 1);
      {
  var slice_coord_TMP : i32= (s_ch) / 4;
  var sub_ch_coord_TMP : i32= (s_ch) % 4;
  result.z = src_tensor_buffer.data[(((slice_coord_TMP) * U.i1.w + (s_y)) * U.i2.x + (s_x))][sub_ch_coord_TMP];
  };
  }
  {
    var s_ch : i32= min((S * 4 + 3) * U.i2.w + U.i1.y, U.i1.z - 1);
      {
  var slice_coord_TMP : i32= (s_ch) / 4;
  var sub_ch_coord_TMP : i32= (s_ch) % 4;
  result.w = src_tensor_buffer.data[(((slice_coord_TMP) * U.i1.w + (s_y)) * U.i2.x + (s_x))][sub_ch_coord_TMP];
  };
  }
  {

   var result_final : vec4<f16>;
  {  
  {result_final = vec4<f16>(result);
  }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.y + (S))), vec4<f32>(result_final));
};
}
