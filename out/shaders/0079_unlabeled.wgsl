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
struct weights_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> weights_buffer : weights_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {
  var linear_spatial : i32= i32(reserved_gid.x);
  var DST_X : i32= linear_spatial % U.i2.y;
  linear_spatial = linear_spatial / U.i2.y;
  var DST_Y : i32= linear_spatial % U.i2.z;
  linear_spatial = linear_spatial / U.i2.z;
  var DST_S : i32= i32(reserved_group_id.y);
  DST_X *= 2;
  DST_S *= 2;
  if (DST_S >= U.i1.y) {return;}
  var w_batch_id : i32= DST_Y;
  if (DST_Y >= U.i1.x || DST_S >= U.i1.y) {
    return;
  }
  var r_w0_h0_s0 : vec4<f16>= vec4<f16>(0., 0., 0., 0.);
  var r_w1_h0_s0 : vec4<f16>= vec4<f16>(0., 0., 0., 0.);
  var r_w0_h0_s1 : vec4<f16>= vec4<f16>(0., 0., 0., 0.);
  var r_w1_h0_s1 : vec4<f16>= vec4<f16>(0., 0., 0., 0.);
  var xc0 : i32= DST_X + 0;
  var xc1 : i32= DST_X + 1;
  var yc0 : i32= DST_Y + 0;
  var weights_offset : i32= DST_S * 4 * U.i2.x;
  var s : i32= 0;
  loop {
    var src_w0_h0 : vec4<f16>;
    var src_w1_h0 : vec4<f16>;
    src_w0_h0 = vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((xc0), ((yc0) * U.i2.x + (s))), 0));
    src_w1_h0 = vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((xc1), ((yc0) * U.i2.x + (s))), 0));
    s += 1;
    r_w0_h0_s0.x += dot(weights_buffer.data[weights_offset + 0], src_w0_h0);
    r_w1_h0_s0.x += dot(weights_buffer.data[weights_offset + 0], src_w1_h0);
    r_w0_h0_s0.y += dot(weights_buffer.data[weights_offset + 1], src_w0_h0);
    r_w1_h0_s0.y += dot(weights_buffer.data[weights_offset + 1], src_w1_h0);
    r_w0_h0_s0.z += dot(weights_buffer.data[weights_offset + 2], src_w0_h0);
    r_w1_h0_s0.z += dot(weights_buffer.data[weights_offset + 2], src_w1_h0);
    r_w0_h0_s0.w += dot(weights_buffer.data[weights_offset + 3], src_w0_h0);
    r_w1_h0_s0.w += dot(weights_buffer.data[weights_offset + 3], src_w1_h0);
    r_w0_h0_s1.x += dot(weights_buffer.data[weights_offset + 4], src_w0_h0);
    r_w1_h0_s1.x += dot(weights_buffer.data[weights_offset + 4], src_w1_h0);
    r_w0_h0_s1.y += dot(weights_buffer.data[weights_offset + 5], src_w0_h0);
    r_w1_h0_s1.y += dot(weights_buffer.data[weights_offset + 5], src_w1_h0);
    r_w0_h0_s1.z += dot(weights_buffer.data[weights_offset + 6], src_w0_h0);
    r_w1_h0_s1.z += dot(weights_buffer.data[weights_offset + 6], src_w1_h0);
    r_w0_h0_s1.w += dot(weights_buffer.data[weights_offset + 7], src_w0_h0);
    r_w1_h0_s1.w += dot(weights_buffer.data[weights_offset + 7], src_w1_h0);
    weights_offset += 8;
  if(! (s < U.i2.x)){break;}
} ;
  if (DST_S + 0 >= U.i1.y) {return;}
  {
    var bias_val : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
    if (DST_S < 0) {
      bias_val = vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i2.x + (0))), 0));
    }
  {
    var res : vec4<f16>= vec4<f16>(r_w0_h0_s0);
    res += bias_val;
    {

   var res_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((((DST_X + 0) % 8)), ((0) * U.i1.w + ((DST_S + 0)))), 0));
  
        res_final.x =  select( res.x,  f16(U.f0.x) , second_value.x );
        res_final.y =  select( res.y,  f16(U.f0.x) , second_value.y );
        res_final.z =  select( res.z,  f16(U.f0.x) , second_value.z );
        res_final.w =  select( res.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((DST_X + 0), ((DST_Y + 0) * U.i1.y + (DST_S + 0))), vec4<f32>(res_final));
};
  }
  if (DST_X + 1 < U.i1.z) {
    var res : vec4<f16>= vec4<f16>(r_w1_h0_s0);
    res += bias_val;
    {

   var res_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((((DST_X + 1) % 8)), ((0) * U.i1.w + ((DST_S + 0)))), 0));
  
        res_final.x =  select( res.x,  f16(U.f0.x) , second_value.x );
        res_final.y =  select( res.y,  f16(U.f0.x) , second_value.y );
        res_final.z =  select( res.z,  f16(U.f0.x) , second_value.z );
        res_final.w =  select( res.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((DST_X + 1), ((DST_Y + 0) * U.i1.y + (DST_S + 0))), vec4<f32>(res_final));
};
  }
  }
  if (DST_S + 1 >= U.i1.y) {return;}
  {
    var bias_val : vec4<f16>= vec4<f16>(0.0, 0.0, 0.0, 0.0);
    if (DST_S < 0) {
      bias_val = vec4<f16>(textureLoad(src_tensor_image2d, vec2<i32>((0), ((0) * U.i2.x + (0))), 0));
    }
  {
    var res : vec4<f16>= vec4<f16>(r_w0_h0_s1);
    res += bias_val;
    {

   var res_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((((DST_X + 0) % 8)), ((0) * U.i1.w + ((DST_S + 1)))), 0));
  
        res_final.x =  select( res.x,  f16(U.f0.x) , second_value.x );
        res_final.y =  select( res.y,  f16(U.f0.x) , second_value.y );
        res_final.z =  select( res.z,  f16(U.f0.x) , second_value.z );
        res_final.w =  select( res.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((DST_X + 0), ((DST_Y + 0) * U.i1.y + (DST_S + 1))), vec4<f32>(res_final));
};
  }
  if (DST_X + 1 < U.i1.z) {
    var res : vec4<f16>= vec4<f16>(r_w1_h0_s1);
    res += bias_val;
    {

   var res_final : vec4<f16>;
  {  
  {var second_value : vec4<bool>= vec4<bool>(textureLoad(src_tensor_1_link1_image2d, vec2<i32>((((DST_X + 1) % 8)), ((0) * U.i1.w + ((DST_S + 1)))), 0));
  
        res_final.x =  select( res.x,  f16(U.f0.x) , second_value.x );
        res_final.y =  select( res.y,  f16(U.f0.x) , second_value.y );
        res_final.z =  select( res.z,  f16(U.f0.x) , second_value.z );
        res_final.w =  select( res.w,  f16(U.f0.x) , second_value.w );
    }
  }
  textureStore(dst_tensor_image2d, vec2<i32>((DST_X + 1), ((DST_Y + 0) * U.i1.y + (DST_S + 1))), vec4<f32>(res_final));
};
  }
  }
}
