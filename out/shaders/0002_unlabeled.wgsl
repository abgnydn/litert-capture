enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct cache_k_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(0) var<storage, read_write> cache_k_buffer : cache_k_buffer_vector;
struct cache_v_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(1) var<storage, read_write> cache_v_buffer : cache_v_buffer_vector;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(2) var<storage, read> params_buffer : params_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
fn QuantizedBufferWrite(src : vec4<u32>) -> u32 {
  var dst: u32;
  dst = insertBits(dst, src.x, 0, 8);
  dst = insertBits(dst, src.y, 8, 8);
  dst = insertBits(dst, src.z, 16, 8);
  dst = insertBits(dst, src.w, 24, 8);
  return dst;
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>) {
  var X : i32= i32(reserved_gid.x);
  var Y : i32= i32(reserved_gid.y);
  var S : i32= i32(reserved_gid.z);
  if (X >= U.i0.y || Y >= U.i0.x || S >= U.i0.w) {
    return;
  }

  var token_index : i32= X;

  
  
  var time_step : i32= params_buffer.data[0];
  if (token_index >= U.i0.z || token_index < time_step) {
    return;
  }
  token_index %= U.i0.y;
  {
    
    
    
    
    
    var k_o : i32= token_index;
    var k_o_slice : i32= k_o / 4;
    var k_o4_index : i32= k_o % 4;
    var k_i : i32= S * 4;
    var k_i_slice : i32= k_i / 4;
    var k_sp : i32= Y;
    var i_slices : i32= (U.i0.w + 3) / 4;
    var o_slices : i32= (U.i0.y + 3) / 4;
    var k_index : i32= ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    cache_k_buffer.data[k_index] = QuantizedBufferWrite(vec4<u32>(0, 0, 0, 0));
  }
  {
    
    
    
    
    
    var v_o : i32= S * 4;
    var v_o_slice : i32= v_o / 4;
    var v_i : i32= token_index;
    var v_i_slice : i32= v_i / 4;
    var v_i4_index : i32= v_i % 4;
    var v_sp : i32= Y;
    var i_slices : i32= (U.i0.y + 3) / 4;
    var o_slices : i32= (U.i0.w + 3) / 4;
    var v_index : i32= ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    cache_v_buffer.data[v_index] = QuantizedBufferWrite(vec4<u32>(0, 0, 0, 0));
  }
}