enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
@group(0) @binding(0) var dst_tensor_image2d : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var src_tensor_image2d : texture_2d<f32>;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(2) var<storage, read> params_buffer : params_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
  i1 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
var<workgroup> loc_mem : array<array<vec2<f32>, 16>, 8>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var sum : f32= 0.0;
  var end_channel : i32= params_buffer.data[U.i0.w];
  var end_slice : i32= (end_channel + 3) / 4;
  var need_per_channels_check : bool= end_channel % 4 != 0;
  var maximum : f32;
    {
  var slice_coord_TMP : i32= (0) / 4;
  var sub_ch_coord_TMP : i32= (0) % 4;
  maximum = textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.x + (slice_coord_TMP))), 0)[sub_ch_coord_TMP];
  };
  for (var d : i32= i32(reserved_lid.z); d < end_slice; d += 8) {
    var mask_dot : vec4<f32>= vec4<f32>(1.0, 1.0, 1.0, 1.0);
    var src : vec4<f32>= textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.x + (d))), 0);
    if (need_per_channels_check && (d == end_slice - 1)) {
      if (d * 4 + 0 >= end_channel) {
        mask_dot.x = 0.0;
        src.x = maximum;
      }
      if (d * 4 + 1 >= end_channel) {
        mask_dot.y = 0.0;
        src.y = maximum;
      }
      if (d * 4 + 2 >= end_channel) {
        mask_dot.z = 0.0;
        src.z = maximum;
      }
      if (d * 4 + 3 >= end_channel) {
        mask_dot.w = 0.0;
        src.w = maximum;
      }
    }
    var new_max : f32= max(src.x, src.y);
    new_max = max(new_max, src.z);
    new_max = max(new_max, src.w);
    new_max = max(new_max, maximum);
    var scale : f32= exp(maximum - new_max);
    maximum = new_max;
    sum *= scale;
    var exp_res : vec4<f32>= exp(src - vec4<f32>(maximum, maximum, maximum, maximum));
    sum += dot(mask_dot, exp_res);
  }
  var value : vec2<f32>;
  value.x = sum;
  value.y = maximum;
  var local_xy : i32= i32(reserved_lid.y) * 16 + i32(reserved_lid.x);
  loc_mem[i32(reserved_lid.z)][local_xy] = value;
  workgroupBarrier();
  if (i32(reserved_lid.z) == 0) {
    {
    var new_value : vec2<f32>= loc_mem[1][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[2][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[3][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[4][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[5][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[6][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    {
    var new_value : vec2<f32>= loc_mem[7][local_xy];
    var new_max : f32= max(value.y, new_value.y);
    new_value.x *= exp(new_value.y - new_max);
    value.x *= exp(value.y - new_max);
    value.x = value.x + new_value.x;
    value.y = new_max;
    }
    sum = value.x;
    maximum = value.y;
  }
  var inv_sum : f32= 1.0 / sum;
  if (i32(reserved_lid.z) != 0) {return;}
  if (X >= U.i0.z || Y >= U.i0.x) {return;} 
  var result : vec4<f16>;
  result.x = f16(inv_sum);
  result.y = f16(maximum);
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i0.y + (0))), vec4<f32>(result));
}
