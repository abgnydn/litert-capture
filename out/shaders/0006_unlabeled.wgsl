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
@group(0) @binding(0) var src_texture_image2d : texture_2d<u32>;
struct dst_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(1) var<storage, read_write> dst_tensor_buffer : dst_tensor_buffer_vector;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(2) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var group_id : i32= i32(reserved_group_id.y) * U.i0.z + i32(reserved_group_id.x);
  var linear_index : i32= group_id * WORKGROUP_SIZE_X + i32(reserved_lid.x);
  if (linear_index >= U.i0.y) {return;}
  if (i32(reserved_gid.z) != 0) {return;}
  var dst_o_sp_i_ogroup : i32= linear_index;
  var dst_ogroup : i32= dst_o_sp_i_ogroup % U.i0.x;
  var dst_o_sp_i : i32= dst_o_sp_i_ogroup / U.i0.x;
  var dst_i : i32= dst_o_sp_i % U.i0.w;
  var dst_o_sp : i32= dst_o_sp_i / U.i0.w;
  var dst_sp : i32= dst_o_sp % U.i1.w;
  var dst_o : i32= dst_o_sp / U.i1.w;
  var i_slice : i32= dst_i;
  var o_slice : i32= dst_o * U.i0.x + dst_ogroup;
  var spatial_linear : i32= dst_sp;
  var W : i32= spatial_linear % U.i2.x;
  var H : i32= spatial_linear / U.i2.x;
  var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
  w0 = vec4<f16>(0, 0, 0, 0);
  w1 = vec4<f16>(0, 0, 0, 0);
  w2 = vec4<f16>(0, 0, 0, 0);
  w3 = vec4<f16>(0, 0, 0, 0);
  if (o_slice < U.i1.x && i_slice < U.i0.w) {
  var src_i : i32= i_slice;
  var src_o_group : i32= o_slice % U.i1.y;
  var src_o : i32= o_slice / U.i1.y;
  var src_linear : i32= ((src_o * U.i1.w + spatial_linear) * U.i0.w + src_i) * U.i1.y + src_o_group;
    var w : vec4<u32>= vec4<u32>(textureLoad(src_texture_image2d, vec2<i32>((src_o_group), (((spatial_linear * U.i0.w + src_i) * U.i1.z + src_o))), 0));
    
  w0.x = f16((w.x) & 15u);
  w0.y = f16((w.x >>  4u) & 15u);
  w0.z = f16((w.x >>  8u) & 15u);
  w0.w = f16((w.x >> 12u) & 15u);
  w1.x = f16((w.y) & 15u);
  w1.y = f16((w.y >> 4u) & 15u);
  w1.z = f16((w.y >> 8u) & 15u);
  w1.w = f16((w.y >> 12u) & 15u);
  w2.x = f16((w.z) & 15u);
  w2.y = f16((w.z >>  4u) & 15u);
  w2.z = f16((w.z >>  8u) & 15u);
  w2.w = f16((w.z >> 12u) & 15u);
  w3.x = f16((w.w) & 15u);
  w3.y = f16((w.w >> 4u) & 15u);
  w3.z = f16((w.w >> 8u) & 15u);
  w3.w = f16((w.w >> 12u) & 15u);
;
  }  
  if (o_slice < U.i1.x && i_slice < U.i0.w) {
    var weight_scale : vec4<f16>= vec4<f16>(weights_scale_buffer.data[(o_slice)]);
    var weight_zp : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var weight_bias : vec4<f16>= -weight_scale * (vec4<f16>(8.0, 8.0, 8.0, 8.0) + weight_zp);
  w0 = w0 * weight_scale + weight_bias;
  w1 = w1 * weight_scale + weight_bias;
  w2 = w2 * weight_scale + weight_bias;
  w3 = w3 * weight_scale + weight_bias;
  }  
  var r0 : vec4<f16>= vec4<f16>(w0.x, w1.x, w2.x, w3.x);
  var r1 : vec4<f16>= vec4<f16>(w0.y, w1.y, w2.y, w3.y);
  var r2 : vec4<f16>= vec4<f16>(w0.z, w1.z, w2.z, w3.z);
  var r3 : vec4<f16>= vec4<f16>(w0.w, w1.w, w2.w, w3.w);
  dst_tensor_buffer.data[linear_index * 4 + 0] = r0;
  dst_tensor_buffer.data[linear_index * 4 + 1] = r1;
  dst_tensor_buffer.data[linear_index * 4 + 2] = r2;
  dst_tensor_buffer.data[linear_index * 4 + 3] = r3;
}
