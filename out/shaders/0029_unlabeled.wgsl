enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var mask_tensor_image2d : texture_storage_2d<rgba8uint, write>;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> params_buffer : params_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
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
  var token_index_offset : i32= params_buffer.data[0];
  var temps : array<bool, 4>;
  for (var i : i32= 0; i < 4; i=i+1) {
    var token_index : i32= S * 4 + i - token_index_offset;
    temps[i] =  select( true,  false , token_index <= X );

  }
  var mask_value : vec4<bool>;
  mask_value.x = temps[0];
  mask_value.y = temps[1];
  mask_value.z = temps[2];
  mask_value.w = temps[3];
  textureStore(mask_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.y + (S))), vec4<u32>(vec4<u32>(mask_value)));
}