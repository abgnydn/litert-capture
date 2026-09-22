enable f16;
@id(0) override WORKGROUP_SIZE_X: i32;
@id(1) override WORKGROUP_SIZE_Y: i32;
@id(2) override WORKGROUP_SIZE_Z: i32;
struct k_local_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(0) var<storage, read_write> k_local_buffer : k_local_buffer_vector;
struct params_buffer_vector {
  data: array<i32>,
};
@group(0) @binding(1) var<storage, read> params_buffer : params_buffer_vector;
struct src_k_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(2) var<storage, read> src_k_buffer : src_k_buffer_vector;
struct src_v_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(3) var<storage, read> src_v_buffer : src_v_buffer_vector;
struct v_local_buffer_vector {
  data: array<u32>,
};
@group(0) @binding(4) var<storage, read_write> v_local_buffer : v_local_buffer_vector;
struct Scalars {
  i0 : vec4<i32>,
};
@group(0) @binding(5) var<uniform> U: Scalars;
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
  var dst_cache_id : i32= i32(reserved_gid.x);
  var kv_id : i32= i32(reserved_gid.y);
  var head_dim_id : i32= i32(reserved_gid.z);
  if (dst_cache_id >= U.i0.z || kv_id >= U.i0.y || head_dim_id >= U.i0.x / 4) {
    return;
  }

  var ring_offset : i32= params_buffer.data[4];
  var src_cache_id : i32= (dst_cache_id + ring_offset) % U.i0.w;

  {
    
    
    
    
    
    var dst_k_o : i32= dst_cache_id;
    var dst_k_o_slice : i32= dst_k_o / 4;
    var dst_k_o4_index : i32= dst_k_o % 4;

    var src_k_o : i32= src_cache_id;
    var src_k_o_slice : i32= src_k_o / 4;
    var src_k_o4_index : i32= src_k_o % 4;

    var k_i : i32= head_dim_id * 4;
    var k_i_slice : i32= k_i / 4;
    var k_sp : i32= kv_id;
    var i_slices : i32= (U.i0.x + 3) / 4;
    var src_o_slices : i32= (U.i0.w + 3) / 4;
    var dst_o_slices : i32= (U.i0.z + 3) / 4;
    var src_index : i32= ((k_sp * i_slices + k_i_slice) * src_o_slices + src_k_o_slice) * 4 + src_k_o4_index;
    var dst_index : i32= ((k_sp * i_slices + k_i_slice) * dst_o_slices + dst_k_o_slice) * 4 + dst_k_o4_index;
    var value_k : vec4<u32>= vec4<u32>(extractBits(src_k_buffer.data[(src_index)], 0, 8), extractBits(src_k_buffer.data[(src_index)], 8, 8), extractBits(src_k_buffer.data[(src_index)], 16, 8), extractBits(src_k_buffer.data[(src_index)], 24, 8));
    k_local_buffer.data[dst_index] = QuantizedBufferWrite(value_k);
  }
  {
    
    
    
    
    
    var v_o : i32= head_dim_id * 4;
    var v_o_slice : i32= v_o / 4;

    var dst_v_i : i32= dst_cache_id;
    var dst_v_i_slice : i32= dst_v_i / 4;
    var dst_v_i4_index : i32= dst_v_i % 4;

    var src_v_i : i32= src_cache_id;
    var src_v_i_slice : i32= src_v_i / 4;
    var src_v_i4_index : i32= src_v_i % 4;

    var v_sp : i32= kv_id;
    var o_slices : i32= (U.i0.x + 3) / 4;
    var src_i_slices : i32= (U.i0.w + 3) / 4;
    var dst_i_slices : i32= (U.i0.z + 3) / 4;
    var src_index : i32= ((v_sp * src_i_slices + src_v_i_slice) * o_slices + v_o_slice) * 4 + src_v_i4_index;
    var dst_index : i32= ((v_sp * dst_i_slices + dst_v_i_slice) * o_slices + v_o_slice) * 4 + dst_v_i4_index;
    var value_v : vec4<u32>= vec4<u32>(extractBits(src_v_buffer.data[(src_index)], 0, 8), extractBits(src_v_buffer.data[(src_index)], 8, 8), extractBits(src_v_buffer.data[(src_index)], 16, 8), extractBits(src_v_buffer.data[(src_index)], 24, 8));
    v_local_buffer.data[dst_index] = QuantizedBufferWrite(value_v);
  }
}