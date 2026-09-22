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
@group(0) @binding(0) var dst_ind_image2d : texture_storage_2d<rgba32sint, write>;
@group(0) @binding(1) var dst_max_image2d : texture_storage_2d<rgba16float, write>;
struct src_buffer_vector {
  data: array<vec4<f16>>,
};
@group(0) @binding(2) var<storage, read> src_buffer : src_buffer_vector;
struct Scalars {
  f0 : vec4<f32>,
  i1 : vec4<i32>,
  i2 : vec4<i32>,
};
@group(0) @binding(3) var<uniform> U: Scalars;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) reserved_gid : vec3<u32>,
@builtin(local_invocation_id) reserved_lid : vec3<u32>,
@builtin(workgroup_id) reserved_group_id : vec3<u32>) {

  var H : i32= i32(reserved_group_id.y);
  var local_id : i32= i32(reserved_lid.x);
  var start_position : i32= i32(reserved_group_id.x) * U.i1.z;
  var end_position : i32= min(start_position + U.i1.z, U.i2.x);

  var top_k_ind_0 : vec4<i32>= vec4<i32>(-1, -1, -1, -1);
  var top_k_max_0 : vec4<f16>= vec4<f16>(f16(U.f0.x), f16(U.f0.x), f16(U.f0.x), f16(U.f0.x));
  for (var w : i32= start_position + local_id; w < end_position; w += WORKGROUP_SIZE_X) {
    var new_value : vec4<f16>= src_buffer.data[(((0) * U.i1.w + (H)) * U.i2.x + (w))];
    var new_inds : vec4<i32>= vec4<i32>(w * 4 + 0, w * 4 + 1, w * 4 + 2, w * 4 + 3);
    var value_to_add : f16= new_value.x;
    var index_to_add : i32= new_inds.x;
      if (value_to_add >= top_k_max_0.x) {
    var tmp_ind : i32= top_k_ind_0.x;
    if (top_k_max_0.x == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.x);
      index_to_add = min(index_to_add, top_k_ind_0.x);
    }
    var tmp_max : f16= top_k_max_0.x;
    top_k_max_0.x = value_to_add;
    top_k_ind_0.x = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.y) {
    var tmp_ind : i32= top_k_ind_0.y;
    if (top_k_max_0.y == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.y);
      index_to_add = min(index_to_add, top_k_ind_0.y);
    }
    var tmp_max : f16= top_k_max_0.y;
    top_k_max_0.y = value_to_add;
    top_k_ind_0.y = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.z) {
    var tmp_ind : i32= top_k_ind_0.z;
    if (top_k_max_0.z == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.z);
      index_to_add = min(index_to_add, top_k_ind_0.z);
    }
    var tmp_max : f16= top_k_max_0.z;
    top_k_max_0.z = value_to_add;
    top_k_ind_0.z = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.w) {
    var tmp_ind : i32= top_k_ind_0.w;
    if (top_k_max_0.w == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.w);
      index_to_add = min(index_to_add, top_k_ind_0.w);
    }
    var tmp_max : f16= top_k_max_0.w;
    top_k_max_0.w = value_to_add;
    top_k_ind_0.w = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }

    value_to_add = new_value.y;
    index_to_add = new_inds.y;
      if (value_to_add >= top_k_max_0.x) {
    var tmp_ind : i32= top_k_ind_0.x;
    if (top_k_max_0.x == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.x);
      index_to_add = min(index_to_add, top_k_ind_0.x);
    }
    var tmp_max : f16= top_k_max_0.x;
    top_k_max_0.x = value_to_add;
    top_k_ind_0.x = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.y) {
    var tmp_ind : i32= top_k_ind_0.y;
    if (top_k_max_0.y == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.y);
      index_to_add = min(index_to_add, top_k_ind_0.y);
    }
    var tmp_max : f16= top_k_max_0.y;
    top_k_max_0.y = value_to_add;
    top_k_ind_0.y = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.z) {
    var tmp_ind : i32= top_k_ind_0.z;
    if (top_k_max_0.z == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.z);
      index_to_add = min(index_to_add, top_k_ind_0.z);
    }
    var tmp_max : f16= top_k_max_0.z;
    top_k_max_0.z = value_to_add;
    top_k_ind_0.z = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.w) {
    var tmp_ind : i32= top_k_ind_0.w;
    if (top_k_max_0.w == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.w);
      index_to_add = min(index_to_add, top_k_ind_0.w);
    }
    var tmp_max : f16= top_k_max_0.w;
    top_k_max_0.w = value_to_add;
    top_k_ind_0.w = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }

    value_to_add = new_value.z;
    index_to_add = new_inds.z;
      if (value_to_add >= top_k_max_0.x) {
    var tmp_ind : i32= top_k_ind_0.x;
    if (top_k_max_0.x == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.x);
      index_to_add = min(index_to_add, top_k_ind_0.x);
    }
    var tmp_max : f16= top_k_max_0.x;
    top_k_max_0.x = value_to_add;
    top_k_ind_0.x = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.y) {
    var tmp_ind : i32= top_k_ind_0.y;
    if (top_k_max_0.y == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.y);
      index_to_add = min(index_to_add, top_k_ind_0.y);
    }
    var tmp_max : f16= top_k_max_0.y;
    top_k_max_0.y = value_to_add;
    top_k_ind_0.y = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.z) {
    var tmp_ind : i32= top_k_ind_0.z;
    if (top_k_max_0.z == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.z);
      index_to_add = min(index_to_add, top_k_ind_0.z);
    }
    var tmp_max : f16= top_k_max_0.z;
    top_k_max_0.z = value_to_add;
    top_k_ind_0.z = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.w) {
    var tmp_ind : i32= top_k_ind_0.w;
    if (top_k_max_0.w == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.w);
      index_to_add = min(index_to_add, top_k_ind_0.w);
    }
    var tmp_max : f16= top_k_max_0.w;
    top_k_max_0.w = value_to_add;
    top_k_ind_0.w = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }

    value_to_add = new_value.w;
    index_to_add = new_inds.w;
      if (value_to_add >= top_k_max_0.x) {
    var tmp_ind : i32= top_k_ind_0.x;
    if (top_k_max_0.x == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.x);
      index_to_add = min(index_to_add, top_k_ind_0.x);
    }
    var tmp_max : f16= top_k_max_0.x;
    top_k_max_0.x = value_to_add;
    top_k_ind_0.x = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.y) {
    var tmp_ind : i32= top_k_ind_0.y;
    if (top_k_max_0.y == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.y);
      index_to_add = min(index_to_add, top_k_ind_0.y);
    }
    var tmp_max : f16= top_k_max_0.y;
    top_k_max_0.y = value_to_add;
    top_k_ind_0.y = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.z) {
    var tmp_ind : i32= top_k_ind_0.z;
    if (top_k_max_0.z == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.z);
      index_to_add = min(index_to_add, top_k_ind_0.z);
    }
    var tmp_max : f16= top_k_max_0.z;
    top_k_max_0.z = value_to_add;
    top_k_ind_0.z = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
  if (value_to_add >= top_k_max_0.w) {
    var tmp_ind : i32= top_k_ind_0.w;
    if (top_k_max_0.w == value_to_add) {
      tmp_ind = max(index_to_add, top_k_ind_0.w);
      index_to_add = min(index_to_add, top_k_ind_0.w);
    }
    var tmp_max : f16= top_k_max_0.w;
    top_k_max_0.w = value_to_add;
    top_k_ind_0.w = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }

  }

  textureStore(dst_max_image2d, vec2<i32>((i32(reserved_gid.x)), ((H) * U.i1.y + (0))), vec4<f32>(top_k_max_0));
  textureStore(dst_ind_image2d, vec2<i32>((i32(reserved_gid.x)), ((H) * U.i1.x + (0))), top_k_ind_0);
}