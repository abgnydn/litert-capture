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
@group(0) @binding(1) var src_tensor_1_link1_image2d : texture_2d<u32>;
@group(0) @binding(2) var src_tensor_image2d : texture_2d<f32>;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(3) var<storage, read> params_buffer : params_buffer_vector;
struct weights_buffer_vector {
  data: array<vec4<u32>>,
};
@group(0) @binding(4) var<storage, read> weights_buffer : weights_buffer_vector;
struct weights_scale_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(5) var<storage, read> weights_scale_buffer : weights_scale_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(6) var<uniform> U: Scalars;
var<workgroup> temp : array<array<vec4<f16>, 64>, 8>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var dst_s : i32= i32(reserved_gid.x);
  var dst_end_slice : i32= min((((params_buffer.data[U.i1.x] + 32 - 1) / 32) * 32) / 4, U.i1.y);
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
  if (dst_s < U.i1.y) {
  var w_scale : vec4<f16>= weights_scale_buffer.data[(dst_s)];
  var w_bias : vec4<f16>= -w_scale * (vec4<f16>(128.0, 128.0, 128.0, 128.0));
  for (var src_s : i32= tid.y; src_s < U.i2.x; src_s += 4) {
    var v0 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i2.x + (src_s))), 0));
    var v1 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((1), ((0) * U.i2.x + (src_s))), 0));
    var v2 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((2), ((0) * U.i2.x + (src_s))), 0));
    var v3 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((3), ((0) * U.i2.x + (src_s))), 0));
    var v4 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((4), ((0) * U.i2.x + (src_s))), 0));
    var v5 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((5), ((0) * U.i2.x + (src_s))), 0));
    var v6 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((6), ((0) * U.i2.x + (src_s))), 0));
    var v7 : vec4<f16>= vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((7), ((0) * U.i2.x + (src_s))), 0));
    var w0 : vec4<f16>;var w1 : vec4<f16>;var w2 : vec4<f16>;var w3 : vec4<f16>;
    var linear_i4o4 : i32= src_s * U.i1.y + dst_s;
    var w : vec4<u32>= weights_buffer.data[linear_i4o4];
    
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
    w0 = w0 * w_scale.x + w_bias.x;
    w1 = w1 * w_scale.y + w_bias.y;
    w2 = w2 * w_scale.z + w_bias.z;
    w3 = w3 * w_scale.w + w_bias.w;
    r_sp0.x += dot(v0, w0);
    r_sp0.y += dot(v0, w1);
    r_sp0.z += dot(v0, w2);
    r_sp0.w += dot(v0, w3);
    r_sp1.x += dot(v1, w0);
    r_sp1.y += dot(v1, w1);
    r_sp1.z += dot(v1, w2);
    r_sp1.w += dot(v1, w3);
    r_sp2.x += dot(v2, w0);
    r_sp2.y += dot(v2, w1);
    r_sp2.z += dot(v2, w2);
    r_sp2.w += dot(v2, w3);
    r_sp3.x += dot(v3, w0);
    r_sp3.y += dot(v3, w1);
    r_sp3.z += dot(v3, w2);
    r_sp3.w += dot(v3, w3);
    r_sp4.x += dot(v4, w0);
    r_sp4.y += dot(v4, w1);
    r_sp4.z += dot(v4, w2);
    r_sp4.w += dot(v4, w3);
    r_sp5.x += dot(v5, w0);
    r_sp5.y += dot(v5, w1);
    r_sp5.z += dot(v5, w2);
    r_sp5.w += dot(v5, w3);
    r_sp6.x += dot(v6, w0);
    r_sp6.y += dot(v6, w1);
    r_sp6.z += dot(v6, w2);
    r_sp6.w += dot(v6, w3);
    r_sp7.x += dot(v7, w0);
    r_sp7.y += dot(v7, w1);
    r_sp7.z += dot(v7, w2);
    r_sp7.w += dot(v7, w3);
  } 
  } 
  temp[0][tid.x * 4 + tid.y] = r_sp0;
  temp[1][tid.x * 4 + tid.y] = r_sp1;
  temp[2][tid.x * 4 + tid.y] = r_sp2;
  temp[3][tid.x * 4 + tid.y] = r_sp3;
  temp[4][tid.x * 4 + tid.y] = r_sp4;
  temp[5][tid.x * 4 + tid.y] = r_sp5;
  temp[6][tid.x * 4 + tid.y] = r_sp6;
  temp[7][tid.x * 4 + tid.y] = r_sp7;
  for (var ystride : i32= 4 / 2; ystride > 0; ystride /= 2) {
    workgroupBarrier();
    if (tid.y < ystride) {
      r_sp0 += temp[0][tid.x * 4 + tid.y + ystride];
      temp[0][tid.x * 4 + tid.y] = r_sp0;
      r_sp1 += temp[1][tid.x * 4 + tid.y + ystride];
      temp[1][tid.x * 4 + tid.y] = r_sp1;
      r_sp2 += temp[2][tid.x * 4 + tid.y + ystride];
      temp[2][tid.x * 4 + tid.y] = r_sp2;
      r_sp3 += temp[3][tid.x * 4 + tid.y + ystride];
      temp[3][tid.x * 4 + tid.y] = r_sp3;
      r_sp4 += temp[4][tid.x * 4 + tid.y + ystride];
      temp[4][tid.x * 4 + tid.y] = r_sp4;
      r_sp5 += temp[5][tid.x * 4 + tid.y + ystride];
      temp[5][tid.x * 4 + tid.y] = r_sp5;
      r_sp6 += temp[6][tid.x * 4 + tid.y + ystride];
      temp[6][tid.x * 4 + tid.y] = r_sp6;
      r_sp7 += temp[7][tid.x * 4 + tid.y + ystride];
      temp[7][tid.x * 4 + tid.y] = r_sp7;
    }
  }
  if (dst_s >= U.i1.y) {return;}
  if (tid.y != 0) {return;}
  {
  if (0 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp0);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((0), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-1 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp1);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((1), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-2 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp2);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((2), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-3 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp3);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((3), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-4 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp4);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((4), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-5 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp5);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((5), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-6 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp6);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((6), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  if (-7 < U.i1.z) {
  var res_value : vec4<f16>= vec4<f16>(r_sp7);
  {

   var res_value_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((0), ((0) * U.i1.w + ((dst_s)))), 0));
  
        res_value_final.x =  select( res_value.x,  f16(U.f0.x) , second_value.x );
        res_value_final.y =  select( res_value.y,  f16(U.f0.x) , second_value.y );
        res_value_final.z =  select( res_value.z,  f16(U.f0.x) , second_value.z );
        res_value_final.w =  select( res_value.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((7), ((0) * U.i1.y + (dst_s))), vec4<f32>(res_value_final));
};
  }
  }
}
