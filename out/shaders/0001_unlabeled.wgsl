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
struct src_k_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(2) var<storage, read> src_k_buffer : src_k_buffer_vector;
struct src_v_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(3) var<storage, read> src_v_buffer : src_v_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(4) var<uniform> U: Scalars;
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
  if (Y >= U.i0.x || S >= U.i0.z) {
    return;
  }
  var token_index : i32= X;
  if (token_index >= U.i0.w || token_index >= U.i0.y) {
    return;
  }

  {
    
    
    
    
    
    var k_o : i32= token_index;
    var k_o_slice : i32= k_o / 4;
    var k_o4_index : i32= k_o % 4;
    var k_i : i32= S * 4;
    var k_i_slice : i32= k_i / 4;
    var k_sp : i32= Y;
    var i_slices : i32= (U.i0.z + 3) / 4;
    var src_o_slices : i32= (U.i0.w + 3) / 4;
    var src_k_index : i32= ((k_sp * i_slices + k_i_slice) * src_o_slices + k_o_slice) * 4 + k_o4_index;
    var dst_o_slices : i32= (U.i0.y + 3) / 4;
    var dst_k_index : i32= ((k_sp * i_slices + k_i_slice) * dst_o_slices + k_o_slice) * 4 + k_o4_index;
    var result : vec4<u32>= vec4<u32>(extractBits(src_k_buffer.data[(src_k_index)], 0, 8), extractBits(src_k_buffer.data[(src_k_index)], 8, 8), extractBits(src_k_buffer.data[(src_k_index)], 16, 8), extractBits(src_k_buffer.data[(src_k_index)], 24, 8));
    cache_k_buffer.data[dst_k_index] = QuantizedBufferWrite(result);
  }
  {
    
    
    
    
    
    var v_o : i32= S * 4;
    var v_o_slice : i32= v_o / 4;
    var v_i : i32= token_index;
    var v_i_slice : i32= v_i / 4;
    var v_i4_index : i32= v_i % 4;
    var v_sp : i32= Y;
    var o_slices : i32= (U.i0.z + 3) / 4;
    var src_i_slices : i32= (U.i0.w + 3) / 4;
    var src_v_index : i32= ((v_sp * src_i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    var dst_i_slices : i32= (U.i0.y + 3) / 4;
    var dst_v_index : i32= ((v_sp * dst_i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    var result : vec4<u32>= vec4<u32>(extractBits(src_v_buffer.data[(src_v_index)], 0, 8), extractBits(src_v_buffer.data[(src_v_index)], 8, 8), extractBits(src_v_buffer.data[(src_v_index)], 16, 8), extractBits(src_v_buffer.data[(src_v_index)], 24, 8));
    cache_v_buffer.data[dst_v_index] = QuantizedBufferWrite(result);
  }
}