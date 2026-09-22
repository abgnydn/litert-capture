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
@group(0) @binding(0) var src_tensor_image2d : texture_2d<f32>;
struct dst_tensor_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(1) var<storage, read_write> dst_tensor_buffer : dst_tensor_buffer_vector;
struct weights_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(2) var<storage, read> weights_buffer : weights_buffer_vector;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var dst_s : i32= i32(reserved_gid.x);
  var dst_end_slice : i32= U.i0.y;
  if (dst_s >= dst_end_slice) {return;}
  var r_sp0 : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
  var w_scale : vec4<f16>= weights_scale_buffer.data[(dst_s)];
  var w_bias : vec4<f16>= -w_scale * (vec4<f16>(2.0, 2.0, 2.0, 2.0));
  for (var src_s : i32= 0; src_s < U.i0.w; src_s += 1) {
    var v0 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i0.w + (src_s))), 0));
    var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
    var linear_i4o4 : i32= src_s * U.i0.y + dst_s;
    var w : u32= weights_buffer.data[linear_i4o4];
      {

  var wt0 : vec4<f16>;
  wt0.x = f16((w) & 255u) * f16(0.25);
  wt0.y = f16((w >>  8u) & 255u) * f16(0.25);
  wt0.z = f16((w >> 16u) & 255u) * f16(0.25);
  wt0.w = f16((w >> 24u) & 255u) * f16(0.25);

  var wt1 : vec4<f16>= floor(wt0);
  w0.x = (wt0.x - wt1.x) * f16(4.0);
  w1.x = (wt0.y - wt1.y) * f16(4.0);
  w2.x = (wt0.z - wt1.z) * f16(4.0);
  w3.x = (wt0.w - wt1.w) * f16(4.0);
  wt0 = wt1 * f16(0.25);
  wt1 = floor(wt0);
  w0.y = (wt0.x - wt1.x) * f16(4.0);
  w1.y = (wt0.y - wt1.y) * f16(4.0);
  w2.y = (wt0.z - wt1.z) * f16(4.0);
  w3.y = (wt0.w - wt1.w) * f16(4.0);
  wt0 = wt1 * f16(0.25);
  wt1 = floor(wt0);
  w0.z = (wt0.x - wt1.x) * f16(4.0);
  w1.z = (wt0.y - wt1.y) * f16(4.0);
  w2.z = (wt0.z - wt1.z) * f16(4.0);
  w3.z = (wt0.w - wt1.w) * f16(4.0);
  w0.w = wt1.x;
  w1.w = wt1.y;
  w2.w = wt1.z;
  w3.w = wt1.w;
  }
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
  {
  var res_value : vec4<f16>= vec4<f16>(r_sp0);
  dst_tensor_buffer.data[(((dst_s) * U.i0.x + (0)) * U.i0.z + (0))] = res_value;
  }
}
