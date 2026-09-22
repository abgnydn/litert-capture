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
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
@group(0) @binding(2) var weights_image2d : texture_2d<u32>;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
var<workgroup> temp : array<vec4<f16>, 64>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var dst_s : i32= i32(reserved_gid.x);
  var dst_end_slice : i32= U.i0.x;
  var dst_s_wg_offset : i32= i32(reserved_group_id.x) * WORKGROUP_SIZE_X;
  if (dst_s_wg_offset >= dst_end_slice) {return;}
  var r_sp0 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var tid : vec2<i32>;
  tid.x = i32(reserved_lid.x);
  tid.y = i32(reserved_lid.y);
  if (dst_s < U.i0.x) {
  var w_scale : vec4<f16>= weights_scale_buffer.data[(dst_s)];
  var w_bias : vec4<f16>= -w_scale * (vec4<f16>(128.0, 128.0, 128.0, 128.0));
  for (var src_s : i32= tid.y; src_s < U.i0.y; src_s += 64) {
    var v0 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i0.y + (src_s))), 0));
    var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
    var w : vec4<u32>= textureLoad(weights_image2d, vec2<i32>((dst_s), (src_s)), 0);
    
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
    w0 = fma(w0, w_scale, w_bias);
    w1 = fma(w1, w_scale, w_bias);
    w2 = fma(w2, w_scale, w_bias);
    w3 = fma(w3, w_scale, w_bias);
    r_sp0 = fma(vec4<f16>(v0.x, v0.x, v0.x, v0.x), w0, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.y, v0.y, v0.y, v0.y), w1, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.z, v0.z, v0.z, v0.z), w2, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.w, v0.w, v0.w, v0.w), w3, r_sp0);
  } 
  } 
  temp[tid.x * 64 + tid.y] = r_sp0;
  for (var ystride : i32= 64 / 2; ystride > 0; ystride /= 2) {
    workgroupBarrier();
    if (tid.y < ystride) {
      r_sp0 += temp[tid.x * 64 + tid.y + ystride];
      temp[tid.x * 64 + tid.y] = r_sp0;
    }
  }
  if (dst_s >= U.i0.x) {return;}
  if (tid.y != 0) {return;}
  {
  var res_value : vec4<f16>= vec4<f16>(r_sp0);
  textureStore(dst_tensor_image2d, vec2<i32>((0), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
}
