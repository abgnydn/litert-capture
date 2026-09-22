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
var<workgroup> temp : array<array<vec4<f16>, 64>, 8>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var dst_s : i32= i32(reserved_gid.x);
  var dst_end_slice : i32= U.i0.x;
  var dst_s_wg_offset : i32= i32(reserved_group_id.x) * WORKGROUP_SIZE_X;
  if (dst_s_wg_offset >= dst_end_slice) {return;}
  var r_sp0 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp1 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp2 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp3 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp4 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp5 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp6 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var r_sp7 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var tid : vec2<i32>;
  tid.x = i32(reserved_lid.x);
  tid.y = i32(reserved_lid.y);
  if (dst_s < U.i0.x) {
  var w_scale : vec4<f16>= weights_scale_buffer.data[(dst_s)];
  var w_bias : vec4<f16>= -w_scale * (vec4<f16>(8.0, 8.0, 8.0, 8.0));
  for (var src_s : i32= tid.y; src_s < U.i0.z; src_s += 16) {
    var v0 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i0.z + (src_s))), 0));
    var v1 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((1), ((0) * U.i0.z + (src_s))), 0));
    var v2 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((2), ((0) * U.i0.z + (src_s))), 0));
    var v3 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((3), ((0) * U.i0.z + (src_s))), 0));
    var v4 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((4), ((0) * U.i0.z + (src_s))), 0));
    var v5 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((5), ((0) * U.i0.z + (src_s))), 0));
    var v6 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((6), ((0) * U.i0.z + (src_s))), 0));
    var v7 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((7), ((0) * U.i0.z + (src_s))), 0));
    var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
    var w : vec4<u32>= vec4<u32>(textureLoad(weights_image2d, vec2<i32>((dst_s), (src_s)), 0));
    
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
    w0 = fma(w0, w_scale, w_bias);
    w1 = fma(w1, w_scale, w_bias);
    w2 = fma(w2, w_scale, w_bias);
    w3 = fma(w3, w_scale, w_bias);
    r_sp0 = fma(vec4<f16>(v0.x, v0.x, v0.x, v0.x), w0, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.y, v0.y, v0.y, v0.y), w1, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.z, v0.z, v0.z, v0.z), w2, r_sp0);
    r_sp0 = fma(vec4<f16>(v0.w, v0.w, v0.w, v0.w), w3, r_sp0);
    r_sp1 = fma(vec4<f16>(v1.x, v1.x, v1.x, v1.x), w0, r_sp1);
    r_sp1 = fma(vec4<f16>(v1.y, v1.y, v1.y, v1.y), w1, r_sp1);
    r_sp1 = fma(vec4<f16>(v1.z, v1.z, v1.z, v1.z), w2, r_sp1);
    r_sp1 = fma(vec4<f16>(v1.w, v1.w, v1.w, v1.w), w3, r_sp1);
    r_sp2 = fma(vec4<f16>(v2.x, v2.x, v2.x, v2.x), w0, r_sp2);
    r_sp2 = fma(vec4<f16>(v2.y, v2.y, v2.y, v2.y), w1, r_sp2);
    r_sp2 = fma(vec4<f16>(v2.z, v2.z, v2.z, v2.z), w2, r_sp2);
    r_sp2 = fma(vec4<f16>(v2.w, v2.w, v2.w, v2.w), w3, r_sp2);
    r_sp3 = fma(vec4<f16>(v3.x, v3.x, v3.x, v3.x), w0, r_sp3);
    r_sp3 = fma(vec4<f16>(v3.y, v3.y, v3.y, v3.y), w1, r_sp3);
    r_sp3 = fma(vec4<f16>(v3.z, v3.z, v3.z, v3.z), w2, r_sp3);
    r_sp3 = fma(vec4<f16>(v3.w, v3.w, v3.w, v3.w), w3, r_sp3);
    r_sp4 = fma(vec4<f16>(v4.x, v4.x, v4.x, v4.x), w0, r_sp4);
    r_sp4 = fma(vec4<f16>(v4.y, v4.y, v4.y, v4.y), w1, r_sp4);
    r_sp4 = fma(vec4<f16>(v4.z, v4.z, v4.z, v4.z), w2, r_sp4);
    r_sp4 = fma(vec4<f16>(v4.w, v4.w, v4.w, v4.w), w3, r_sp4);
    r_sp5 = fma(vec4<f16>(v5.x, v5.x, v5.x, v5.x), w0, r_sp5);
    r_sp5 = fma(vec4<f16>(v5.y, v5.y, v5.y, v5.y), w1, r_sp5);
    r_sp5 = fma(vec4<f16>(v5.z, v5.z, v5.z, v5.z), w2, r_sp5);
    r_sp5 = fma(vec4<f16>(v5.w, v5.w, v5.w, v5.w), w3, r_sp5);
    r_sp6 = fma(vec4<f16>(v6.x, v6.x, v6.x, v6.x), w0, r_sp6);
    r_sp6 = fma(vec4<f16>(v6.y, v6.y, v6.y, v6.y), w1, r_sp6);
    r_sp6 = fma(vec4<f16>(v6.z, v6.z, v6.z, v6.z), w2, r_sp6);
    r_sp6 = fma(vec4<f16>(v6.w, v6.w, v6.w, v6.w), w3, r_sp6);
    r_sp7 = fma(vec4<f16>(v7.x, v7.x, v7.x, v7.x), w0, r_sp7);
    r_sp7 = fma(vec4<f16>(v7.y, v7.y, v7.y, v7.y), w1, r_sp7);
    r_sp7 = fma(vec4<f16>(v7.z, v7.z, v7.z, v7.z), w2, r_sp7);
    r_sp7 = fma(vec4<f16>(v7.w, v7.w, v7.w, v7.w), w3, r_sp7);
  } 
  } 
  temp[0][tid.x * 16 + tid.y] = r_sp0;
  temp[1][tid.x * 16 + tid.y] = r_sp1;
  temp[2][tid.x * 16 + tid.y] = r_sp2;
  temp[3][tid.x * 16 + tid.y] = r_sp3;
  temp[4][tid.x * 16 + tid.y] = r_sp4;
  temp[5][tid.x * 16 + tid.y] = r_sp5;
  temp[6][tid.x * 16 + tid.y] = r_sp6;
  temp[7][tid.x * 16 + tid.y] = r_sp7;
  for (var ystride : i32= 16 / 2; ystride > 0; ystride /= 2) {
    workgroupBarrier();
    if (tid.y < ystride) {
      r_sp0 += temp[0][tid.x * 16 + tid.y + ystride];
      temp[0][tid.x * 16 + tid.y] = r_sp0;
      r_sp1 += temp[1][tid.x * 16 + tid.y + ystride];
      temp[1][tid.x * 16 + tid.y] = r_sp1;
      r_sp2 += temp[2][tid.x * 16 + tid.y + ystride];
      temp[2][tid.x * 16 + tid.y] = r_sp2;
      r_sp3 += temp[3][tid.x * 16 + tid.y + ystride];
      temp[3][tid.x * 16 + tid.y] = r_sp3;
      r_sp4 += temp[4][tid.x * 16 + tid.y + ystride];
      temp[4][tid.x * 16 + tid.y] = r_sp4;
      r_sp5 += temp[5][tid.x * 16 + tid.y + ystride];
      temp[5][tid.x * 16 + tid.y] = r_sp5;
      r_sp6 += temp[6][tid.x * 16 + tid.y + ystride];
      temp[6][tid.x * 16 + tid.y] = r_sp6;
      r_sp7 += temp[7][tid.x * 16 + tid.y + ystride];
      temp[7][tid.x * 16 + tid.y] = r_sp7;
    }
  }
  if (dst_s >= U.i0.x) {return;}
  if (tid.y != 0) {return;}
  {
  if (0 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp0);
  textureStore(dst_tensor_image2d, vec2<i32>((0), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-1 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp1);
  textureStore(dst_tensor_image2d, vec2<i32>((1), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-2 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp2);
  textureStore(dst_tensor_image2d, vec2<i32>((2), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-3 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp3);
  textureStore(dst_tensor_image2d, vec2<i32>((3), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-4 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp4);
  textureStore(dst_tensor_image2d, vec2<i32>((4), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-5 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp5);
  textureStore(dst_tensor_image2d, vec2<i32>((5), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-6 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp6);
  textureStore(dst_tensor_image2d, vec2<i32>((6), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  if (-7 < U.i0.y) {
  var res_value : vec4<f16>= vec4<f16>(r_sp7);
  textureStore(dst_tensor_image2d, vec2<i32>((7), ((0) * U.i0.x + (dst_s))), vec4<f32>(res_value));
  }
  }
}
