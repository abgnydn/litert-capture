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
@group(0) @binding(1) var src_tensor_1_link2_image2d : texture_2d<f32>;
@group(0) @binding(2) var src_tensor_image2d : texture_2d<f32>;
struct second_tensor_link1_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(3) var<storage, read> second_tensor_link1_buffer : second_tensor_link1_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
var<workgroup> shared_mem : array<array<array<vec2<f32>, 16>, 4>, 1>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>) {
var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var private_sum4_sq : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var local_id : i32= i32(reserved_lid.z);
  var reduction_group_size : i32= WORKGROUP_SIZE_Z;

  var private_sum4 : vec4<f32>= vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var S : i32= local_id; S < U.i2.y; S += reduction_group_size) {
    var x_clamped : i32= min(X, U.i2.z - 1);
    var y_clamped : i32= min(Y, U.i2.x - 1);
    var t : vec4<f32>= textureLoad(src_tensor_image2d, vec2<i32>((x_clamped), ((y_clamped) * U.i2.y + (S))), 0);
    private_sum4 += t;
    private_sum4_sq += t * t;
  }
  var sum : vec2<f32>;
  sum.x = dot(private_sum4, vec4<f32>(1.0, 1.0, 1.0, 1.0));
  sum.y = dot(private_sum4_sq, vec4<f32>(1.0, 1.0, 1.0, 1.0));

  {  
    shared_mem[i32(reserved_lid.y)][i32(reserved_lid.x)][local_id] = sum;
    workgroupBarrier();
    
    var reduction_size : i32= 16;
    while (reduction_size > 1) {
      var active_thread_limit : i32= reduction_size / 2;
      var offset : i32= (reduction_size + 1) / 2;
      if (local_id < active_thread_limit) {
        sum += shared_mem[i32(reserved_lid.y)][i32(reserved_lid.x)][local_id + offset];
        shared_mem[i32(reserved_lid.y)][i32(reserved_lid.x)][local_id] = sum;
      }
      workgroupBarrier();
      reduction_size = offset;
    }
    sum = shared_mem[i32(reserved_lid.y)][i32(reserved_lid.x)][0];
  }
  var mean : f32= sum.x * U.f0.x;
  var mean_sq : f32= sum.y * U.f0.x;
  var variance : f32= max(0.0, mean_sq - mean * mean);
  
  if (X >= U.i1.z) { return; }
  if (Y >= U.i1.x) { return; }
  var stddev_inv : f32= inverseSqrt(mean_sq + U.f0.z);

  for (var S : i32= local_id; S < U.i2.y; S += reduction_group_size) {
    var t : vec4<f32>= textureLoad(src_tensor_image2d, vec2<i32>((X), ((Y) * U.i2.y + (S))), 0);

    var t_normalized : vec4<f32>= t * stddev_inv;

    var result : vec4<f16>= vec4<f16>(t_normalized);
    {

   var result_final : vec4<f16>;
  {  
  
   var interm_value_link2 : vec4<f16>;
  {var second_val : vec4<f16>= second_tensor_link1_buffer.data[((S))];
  interm_value_link2 = result * second_val;}
  
   var interm_value_link3 : vec4<f16>;
  {var second_value : vec4<f16>= vec4<f16>(textureLoad(src_tensor_1_link2_image2d, vec2<i32>(((X)), (((Y)) * U.i1.w + ((S)))), 0));
  interm_value_link3 = interm_value_link2 + second_value;}
  {var second_val : vec4<f16>= vec4<f16>(f16(U.f0.y), f16(U.f0.y), f16(U.f0.y), f16(U.f0.y));
  result_final = interm_value_link3 * second_val;}
  }
  textureStore(dst_tensor_image2d, vec2<i32>((X), ((Y) * U.i1.y + (S))), vec4<f32>(result_final));
};

  }
}