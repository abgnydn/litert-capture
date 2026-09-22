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
struct dst_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(0) var<storage, read_write> dst_tensor_buffer : dst_tensor_buffer_vector;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> params_buffer : params_buffer_vector;
struct src_buffer_buffer_vector {
  data: array<vec4<u32>>,
};
@group(0) @binding(2) var<storage, read> src_buffer_buffer : src_buffer_buffer_vector;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var group_id : i32= i32(reserved_group_id.y) * U.i0.w + i32(reserved_group_id.x);
  var linear_index : i32= group_id * WORKGROUP_SIZE_X + i32(reserved_lid.x);
  if (linear_index >= U.i0.z) {return;}
  if (i32(reserved_gid.z) != 0) {return;}
  var dst_o_sp_i_ogroup : i32= linear_index;
  var dst_ogroup : i32= dst_o_sp_i_ogroup % U.i0.y;
  var dst_o_sp_i : i32= dst_o_sp_i_ogroup / U.i0.y;
  var dst_i : i32= dst_o_sp_i % U.i1.x;
  var dst_o_sp : i32= dst_o_sp_i / U.i1.x;
  var dst_sp : i32= dst_o_sp % U.i1.w;
  var dst_o : i32= dst_o_sp / U.i1.w;
  var i_slice : i32= dst_i;
  var o_slice : i32= dst_o * U.i0.y + dst_ogroup;
  var spatial_linear : i32= dst_sp;
  var W : i32= spatial_linear % U.i2.x;
  var H : i32= spatial_linear / U.i2.x;
  var dst_end_slice_runtime : i32= min((((params_buffer.data[U.i0.x] + 32 - 1) / 32) * 32) / 4, U.i1.y);
  if (o_slice >= dst_end_slice_runtime) {return;}
  var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
  w0 = vec4<f16>(0, 0, 0, 0);
  w1 = vec4<f16>(0, 0, 0, 0);
  w2 = vec4<f16>(0, 0, 0, 0);
  w3 = vec4<f16>(0, 0, 0, 0);
  if (o_slice < U.i1.y && i_slice < U.i1.x) {
  var src_i : i32= i_slice;
  var src_o_group : i32= o_slice % U.i1.z;
  var src_o : i32= o_slice / U.i1.z;
  var src_linear : i32= ((src_o * U.i1.w + spatial_linear) * U.i1.x + src_i) * U.i1.z + src_o_group;
    var w : vec4<u32>= src_buffer_buffer.data[src_linear];
    
  w0.x = f16((w.x) & 255u);
  w0.y = f16((w.x >>  8u) & 255u);
  w0.z = f16((w.x >> 16u) & 255u);
  w0.w = f16((w.x >> 24u) & 255u);
  w1.x = f16((w.y) & 255u);
  w1.y = f16((w.y >>  8u) & 255u);
  w1.z = f16((w.y >> 16u) & 255u);
  w1.w = f16((w.y >> 24u) & 255u);
  w2.x = f16((w.z) & 255u);
  w2.y = f16((w.z >>  8u) & 255u);
  w2.z = f16((w.z >> 16u) & 255u);
  w2.w = f16((w.z >> 24u) & 255u);
  w3.x = f16((w.w) & 255u);
  w3.y = f16((w.w >>  8u) & 255u);
  w3.z = f16((w.w >> 16u) & 255u);
  w3.w = f16((w.w >> 24u) & 255u);
;
  }  
  if (o_slice < U.i1.y && i_slice < U.i1.x) {
    var weight_scale : vec4<f16>= vec4<f16>(weights_scale_buffer.data[(o_slice)]);
    var weight_zp : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var weight_bias : vec4<f16>= -weight_scale * (vec4<f16>(128.0, 128.0, 128.0, 128.0) + weight_zp);
  w0 = w0 * weight_scale.x + weight_bias.x;
  w1 = w1 * weight_scale.y + weight_bias.y;
  w2 = w2 * weight_scale.z + weight_bias.z;
  w3 = w3 * weight_scale.w + weight_bias.w;
  }  
  var r0 : vec4<f16>= w0;
  var r1 : vec4<f16>= w1;
  var r2 : vec4<f16>= w2;
  var r3 : vec4<f16>= w3;
  dst_tensor_buffer.data[linear_index * 4 + 0] = r0;
  dst_tensor_buffer.data[linear_index * 4 + 1] = r1;
  dst_tensor_buffer.data[linear_index * 4 + 2] = r2;
  dst_tensor_buffer.data[linear_index * 4 + 3] = r3;
}
