enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var probabilities_image2d : texture_2d<f32>;
@group(0) @binding(1) var src_indices_image2d : texture_2d<i32>;
@group(0) @binding(2) var src_logits_image2d : texture_2d<f32>;
struct output_buffer_vector {
  data: array<vec4<i32>>,
};
@group(0) @binding(3) var<storage, read_write> output_buffer : output_buffer_vector;
struct params_i32_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(4) var<storage, read> params_i32_buffer : params_i32_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(5) var<uniform> U: Scalars;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var BW : i32= i32(reserved_gid.x);
  var H : i32= i32(reserved_gid.y);
  var B : i32= BW % 1;
  var W : i32= BW / 1;
  if (W >= U.i0.y || H >= U.i0.x || i32(reserved_gid.z) != 0) {return;}

  ;
  ;
  ;
  ;

  var probability : f32= textureLoad(probabilities_image2d, vec2<i32>((W), ((H) * U.i0.z + (0))), 0).x;
  var cum_sum : f32= 0.0;
  var index : i32= -1;
  var top_k : i32= params_i32_buffer.data[3];
  for (var s : i32= 0; s < U.i1.x; s=s+1) {
    var vals : vec4<f32>= vec4<f32>(vec4<f16>(textureLoad(src_logits_image2d, vec2<i32>((W), ((H) * U.i1.x + (s))), 0)));
    var inds : vec4<i32>= textureLoad(src_indices_image2d, vec2<i32>((W), ((H) * U.i0.w + (s))), 0);
    if (probability >= cum_sum && s * 4 + 0 < top_k) { index = inds.x; }
    cum_sum += vals.x;
    if (probability >= cum_sum && s * 4 + 1 < top_k) { index = inds.y; }
    cum_sum += vals.y;
    if (probability >= cum_sum && s * 4 + 2 < top_k) { index = inds.z; }
    cum_sum += vals.z;
    if (probability >= cum_sum && s * 4 + 3 < top_k) { index = inds.w; }
    cum_sum += vals.w;
  }
  var result : vec4<i32>= vec4<i32>(index, 0, 0, 0);
  {

   var result_final : vec4<i32>;
  {  
  {result_final = result;}
  }
  output_buffer.data[(((0) * U.i0.x + (H)) * U.i0.y + (W))] = result_final;
};
}