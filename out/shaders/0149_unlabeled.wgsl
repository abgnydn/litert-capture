enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(2) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.w || Y >= U.i0.y || S >= U.i0.z) { 
    return; 
  } 
  var temps : array<f32, 4>;
  temps[0] = f32(0.);
  temps[1] = f32(0.);
  temps[2] = f32(0.);
  temps[3] = f32(0.);
  for (var i : i32= 0; i < 4; i=i+1) {
    var dst_channel : i32= S * 4 + i;
    if (dst_channel < U.i0.x) {
      var s_y : i32= Y;
      var s_x : i32= X;
      var s_c : i32= 0;
        {
  var slice_coord_TMP : i32= (s_c) / 4;
  var sub_ch_coord_TMP : i32= (s_c) % 4;
  temps[i] = textureLoad(src_tensor_image2d, vec2<i32>((s_x), ((s_y) * U.i1.x + (slice_coord_TMP))), 0)[sub_ch_coord_TMP];
  };
    }
  }
  var result : vec4<f32>;
  result.x = temps[0];
  result.y = temps[1];
  result.z = temps[2];
  result.w = temps[3];
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.z + (S))), result);
}
