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
@group(0) @binding(0) var k_dst_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var k_src_image2d : texture_2d<f32>;
@group(0) @binding(2) var position_image2d : texture_2d<i32>;
@group(0) @binding(3) var q_dst_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var q_src_image2d : texture_2d<f32>;
@group(0) @binding(5) var v_dst_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var v_src_image2d : texture_2d<f32>;
struct k_gamma_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(7) var<storage, read> k_gamma_buffer : k_gamma_buffer_vector;
struct q_gamma_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(8) var<storage, read> q_gamma_buffer : q_gamma_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
  i3 : vec4<i32>,
  i4 : vec4<i32>,
};
@group(0) @binding(9) var<uniform> U: Scalars;
var<workgroup> shared_q_sum : array<f32, 64>;
var<workgroup> shared_k_sum : array<f32, 64>;
var<workgroup> shared_inv_std_q : f32;
var<workgroup> shared_inv_std_k : f32;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>) {
  var DST_X : i32= i32(reserved_gid.x);
  var DST_Y : i32= i32(reserved_gid.y);
  var DST_S_GLOBAL : i32= i32(reserved_gid.z);
  var LOCAL_S : i32= i32(reserved_lid.z);

  var half_slices_count : i32= U.i3.x / 2;
  var is_active : bool= (DST_X < U.i3.y && DST_Y < U.i2.w && DST_S_GLOBAL < half_slices_count);

  var DST_S0 : i32= DST_S_GLOBAL;
  var DST_S1 : i32= DST_S_GLOBAL + half_slices_count;

  var batch_id : i32= DST_Y / U.i2.x;
  var kv_head_id : i32= DST_Y % U.i2.x;
  var kv_head_group_id : i32= DST_X / U.i3.w;
  var seq_id : i32= DST_X % U.i3.w;

  ;
  ;
  ;

  var q_val0 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var q_val1 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var local_sum_q : f32= 0.0;
  if (is_active) {
    var q_head_id : i32= kv_head_id * U.i1.w + kv_head_group_id;
    var SRC_S0 : i32= q_head_id * U.i3.x + DST_S0;
    var SRC_S1 : i32= q_head_id * U.i3.x + DST_S1;
    q_val0 = vec4<f32>(vec4<f16>(textureLoad(q_src_image2d, vec2<i32>((seq_id), ((0) * U.i3.z + (SRC_S0))), 0)));
    q_val1 = vec4<f32>(vec4<f16>(textureLoad(q_src_image2d, vec2<i32>((seq_id), ((0) * U.i3.z + (SRC_S1))), 0)));
    local_sum_q = dot(q_val0, q_val0) + dot(q_val1, q_val1);
  }
  shared_q_sum[LOCAL_S] = local_sum_q;

  
  var raw_k_val0 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var raw_k_val1 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var k_val0 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var k_val1 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var local_sum_k : f32= 0.0;
  if (is_active && kv_head_group_id == 0) {
    var K_SRC_S0 : i32= kv_head_id * U.i3.x + DST_S0;
    var K_SRC_S1 : i32= kv_head_id * U.i3.x + DST_S1;
    raw_k_val0 = vec4<f32>(vec4<f16>(textureLoad(k_src_image2d, vec2<i32>((seq_id), ((0) * U.i1.z + (K_SRC_S0))), 0)));
    raw_k_val1 = vec4<f32>(vec4<f16>(textureLoad(k_src_image2d, vec2<i32>((seq_id), ((0) * U.i1.z + (K_SRC_S1))), 0)));
    k_val0 = raw_k_val0;
    k_val1 = raw_k_val1;
    local_sum_k = dot(k_val0, k_val0) + dot(k_val1, k_val1);
  }
  shared_k_sum[LOCAL_S] = local_sum_k;

  workgroupBarrier();

  var reduction_size : i32= 64;
  while (reduction_size > 1) {
    var active_thread_limit : i32= reduction_size / 2;
    var offset : i32= (reduction_size + 1) / 2;
    if (LOCAL_S < active_thread_limit) {
      shared_q_sum[LOCAL_S] += shared_q_sum[LOCAL_S + offset];
      shared_k_sum[LOCAL_S] += shared_k_sum[LOCAL_S + offset];
    }
    workgroupBarrier();
    reduction_size = offset;
  }

  if (LOCAL_S == 0) {
    shared_inv_std_q = inverseSqrt(shared_q_sum[0] / f32(U.i2.z) + 1e-6);

    shared_inv_std_k = inverseSqrt(shared_k_sum[0] / f32(U.i1.x) + 1e-6);

  }
  workgroupBarrier();

  if (is_active) {
    var stddev_inv_q : f32= shared_inv_std_q;
    var val0 : vec4<f32>= q_val0 * stddev_inv_q;
    var val1 : vec4<f32>= q_val1 * stddev_inv_q;

    var gamma0 : vec4<f32>= vec4<f32>(q_gamma_buffer.data[(DST_S0)]);
    var gamma1 : vec4<f32>= vec4<f32>(q_gamma_buffer.data[(DST_S1)]);
    val0 = val0 * gamma0;
    val1 = val1 * gamma1;

    
    var fraction : vec4<f32>;
    var inv_q_dst_ch : f32= 1.0 / f32(U.i2.z);
    fraction.x = 2.0 * f32(DST_S_GLOBAL * 4 + 0) * inv_q_dst_ch;
    fraction.y = 2.0 * f32(DST_S_GLOBAL * 4 + 1) * inv_q_dst_ch;
    fraction.z = 2.0 * f32(DST_S_GLOBAL * 4 + 2) * inv_q_dst_ch;
    fraction.w = 2.0 * f32(DST_S_GLOBAL * 4 + 3) * inv_q_dst_ch;
    var min_timescale : vec4<f32>= vec4<f32>(U.f0.y, U.f0.y, U.f0.y, U.f0.y);
    var max_timescale : vec4<f32>= vec4<f32>(U.f0.x, U.f0.x, U.f0.x, U.f0.x);
    var timescale : vec4<f32>= min_timescale * pow(max_timescale / min_timescale, fraction);
    var pos_val : vec4<f32>= vec4<f32>(vec4<f32>(textureLoad(position_image2d, vec2<i32>((seq_id), ((0) * U.i2.y + (0))), 0)).x, vec4<f32>(textureLoad(position_image2d, vec2<i32>((seq_id), ((0) * U.i2.y + (0))), 0)).x, vec4<f32>(textureLoad(position_image2d, vec2<i32>((seq_id), ((0) * U.i2.y + (0))), 0)).x, vec4<f32>(textureLoad(position_image2d, vec2<i32>((seq_id), ((0) * U.i2.y + (0))), 0)).x);
    var sinusoid_inp : vec4<f32>= pos_val / timescale;
    var sin_val : vec4<f32>= sin(sinusoid_inp);
    var cos_val : vec4<f32>= cos(sinusoid_inp);

    
    {
      var out0 : vec4<f32>;
      var out1 : vec4<f32>;
      if (fraction.w < f32(U.f0.z)) {
        out0 = val0 * cos_val - val1 * sin_val;
        out1 = val1 * cos_val + val0 * sin_val;
      } else {
        out0 = val0;
        out1 = val1;
      }
      var q_out0 : vec4<f16>= vec4<f16>(out0);
      var q_out1 : vec4<f16>= vec4<f16>(out1);
      textureStore(q_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i3.x + (DST_S0))), vec4<f32>(q_out0));
      textureStore(q_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i3.x + (DST_S1))), vec4<f32>(q_out1));
    }

    
    if (kv_head_group_id == 0) {
      var K_SRC_S0 : i32= kv_head_id * U.i3.x + DST_S0;
      var K_SRC_S1 : i32= kv_head_id * U.i3.x + DST_S1;

      
      {
        var stddev_inv_k : f32= shared_inv_std_k;
        k_val0 = k_val0 * stddev_inv_k;
        k_val1 = k_val1 * stddev_inv_k;

        var gamma0 : vec4<f32>= vec4<f32>(k_gamma_buffer.data[(DST_S0)]);
        var gamma1 : vec4<f32>= vec4<f32>(k_gamma_buffer.data[(DST_S1)]);
        k_val0 = k_val0 * gamma0;
        k_val1 = k_val1 * gamma1;

        var out0 : vec4<f32>;
        var out1 : vec4<f32>;
        if (fraction.w < f32(U.f0.z)) {
          out0 = k_val0 * cos_val - k_val1 * sin_val;
          out1 = k_val1 * cos_val + k_val0 * sin_val;
        } else {
          out0 = k_val0;
          out1 = k_val1;
        }
        var k_out0 : vec4<f16>= vec4<f16>(out0);
        var k_out1 : vec4<f16>= vec4<f16>(out1);
        textureStore(k_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i1.y + (DST_S0))), vec4<f32>(k_out0));
        textureStore(k_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i1.y + (DST_S1))), vec4<f32>(k_out1));
      }

      
      {
        var v_out0 : vec4<f16>= vec4<f16>(textureLoad(v_src_image2d, vec2<i32>((seq_id), ((0) * U.i4.y + (K_SRC_S0))), 0));
        var v_out1 : vec4<f16>= vec4<f16>(textureLoad(v_src_image2d, vec2<i32>((seq_id), ((0) * U.i4.y + (K_SRC_S1))), 0));
        textureStore(v_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i4.x + (DST_S0))), vec4<f32>(v_out0));
        textureStore(v_dst_image2d, vec2<i32>((DST_X), ((DST_Y) * U.i4.x + (DST_S1))), vec4<f32>(v_out1));
      }

    }

  }
}