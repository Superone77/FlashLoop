// Adapted from KIVI's MIT-licensed outer-dimension GEMV.  See KIVI_NOTICE.
// FlashLoop adds an explicit logical output width so nested compact deltas do
// not need to be padded to a complete 64-row quantization group.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <torch/extension.h>

#include "kivi_outer_gemv_cuda.h"

namespace {

constexpr int kPackFactor = 8;
constexpr int kWarpSize = 32;
constexpr int kInputsPerThread = 4;
constexpr int kInputTile = kWarpSize * kInputsPerThread;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ void kivi_outer_gemv4_logical_kernel(
    const half* __restrict__ inputs,
    const uint32_t* __restrict__ weights,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ outputs,
    int input_rows,
    int input_channels,
    int output_channels,
    int packed_output_channels,
    int output_groups,
    int group_size,
    int num_heads,
    int num_kv_heads) {
  const int batch_head = blockIdx.x;
  const int input_row = blockIdx.z;
  const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (packed_oc_idx >= packed_output_channels) {
    return;
  }
  const int oc_start = packed_oc_idx * kPackFactor;
  const int group_idx = oc_start / group_size;
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const half* input =
      inputs + (batch_head * input_rows + input_row) * input_channels;
  half* output =
      outputs + (batch_head * input_rows + input_row) * output_channels;
  const half* zero = zeros + kv_batch_head * output_groups * input_channels;
  const half* scale = scales + kv_batch_head * output_groups * input_channels;

  float partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f};
  for (int tile = 0; tile < (input_channels + kInputTile - 1) / kInputTile;
       ++tile) {
#pragma unroll
    for (int lane_item = 0; lane_item < kInputsPerThread; ++lane_item) {
      const int input_index =
          tile * kInputTile + threadIdx.x * kInputsPerThread + lane_item;
      if (input_index < input_channels) {
        uint32_t packed =
            weights[(kv_batch_head * packed_output_channels + packed_oc_idx) *
                        input_channels +
                    input_index];
        const float input_value = __half2float(input[input_index]);
        const float row_scale = __half2float(scale[group_idx * input_channels + input_index]);
        const float row_zero = __half2float(zero[group_idx * input_channels + input_index]);
#pragma unroll
        for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
          const float code = static_cast<float>(packed & 0x0F);
          partial[packed_index] +=
              (row_scale * code + row_zero) * input_value;
          packed >>= 4;
        }
      }
    }
  }

#pragma unroll
  for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
    const int output_channel = oc_start + packed_index;
    const float reduced = warp_reduce_sum(partial[packed_index]);
    if (threadIdx.x == 0 && output_channel < output_channels) {
      output[output_channel] = __float2half(reduced);
    }
  }
}

__global__ void kivi_qk_selected4_kernel(
    const half* __restrict__ inputs,
    const uint32_t* __restrict__ weights,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    const int32_t* __restrict__ indices,
    half* __restrict__ outputs,
    int input_channels,
    int selected_tokens,
    int logical_tokens,
    int packed_tokens,
    int token_groups,
    int group_size,
    int num_heads,
    int num_kv_heads) {
  const int batch_head = blockIdx.x;
  const int selected = blockIdx.y;
  const int logical = indices[batch_head * selected_tokens + selected];
  if (logical < 0 || logical >= logical_tokens) {
    if (threadIdx.x == 0) {
      outputs[batch_head * selected_tokens + selected] = __float2half(0.0f);
    }
    return;
  }
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const int packed_row = logical / kPackFactor;
  const int shift = (logical % kPackFactor) * 4;
  const int group = logical / group_size;
  float partial = 0.0f;
  for (int channel = threadIdx.x; channel < input_channels; channel += kWarpSize) {
    const uint32_t word =
        weights[(kv_batch_head * packed_tokens + packed_row) * input_channels + channel];
    const float code = static_cast<float>((word >> shift) & 0x0F);
    const float scale = __half2float(
        scales[(kv_batch_head * token_groups + group) * input_channels + channel]);
    const float zero = __half2float(
        zeros[(kv_batch_head * token_groups + group) * input_channels + channel]);
    partial += __half2float(inputs[batch_head * input_channels + channel]) *
               (scale * code + zero);
  }
  const float reduced = warp_reduce_sum(partial);
  if (threadIdx.x == 0) {
    outputs[batch_head * selected_tokens + selected] = __float2half(reduced);
  }
}

__global__ void kivi_pv_selected4_kernel(
    const half* __restrict__ inputs,
    const uint32_t* __restrict__ weights,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    const int32_t* __restrict__ indices,
    half* __restrict__ outputs,
    int selected_tokens,
    int logical_tokens,
    int output_channels,
    int packed_output_channels,
    int output_groups,
    int group_size,
    int num_heads,
    int num_kv_heads) {
  const int batch_head = blockIdx.x;
  const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (packed_oc_idx >= packed_output_channels) {
    return;
  }
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const int oc_start = packed_oc_idx * kPackFactor;
  const int group = oc_start / group_size;
  float partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f};
  for (int selected = threadIdx.x; selected < selected_tokens; selected += kWarpSize) {
    const int logical = indices[batch_head * selected_tokens + selected];
    if (logical < 0 || logical >= logical_tokens) {
      continue;
    }
    uint32_t word =
        weights[(kv_batch_head * packed_output_channels + packed_oc_idx) *
                    logical_tokens +
                logical];
    const float input = __half2float(inputs[batch_head * selected_tokens + selected]);
    const float scale = __half2float(
        scales[(kv_batch_head * output_groups + group) * logical_tokens + logical]);
    const float zero = __half2float(
        zeros[(kv_batch_head * output_groups + group) * logical_tokens + logical]);
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const float code = static_cast<float>(word & 0x0F);
      partial[packed_index] += input * (scale * code + zero);
      word >>= 4;
    }
  }
#pragma unroll
  for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
    const int output_channel = oc_start + packed_index;
    const float reduced = warp_reduce_sum(partial[packed_index]);
    if (threadIdx.x == 0 && output_channel < output_channels) {
      outputs[batch_head * output_channels + output_channel] = __float2half(reduced);
    }
  }
}

__global__ void kivi_qk_cross_loop4_kernel(
    const half* __restrict__ inputs,
    const uint32_t* __restrict__ weights0,
    const uint32_t* __restrict__ weights1,
    const uint32_t* __restrict__ weights2,
    const uint32_t* __restrict__ weights3,
    const half* __restrict__ scales0,
    const half* __restrict__ scales1,
    const half* __restrict__ scales2,
    const half* __restrict__ scales3,
    const half* __restrict__ zeros0,
    const half* __restrict__ zeros1,
    const half* __restrict__ zeros2,
    const half* __restrict__ zeros3,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ rank3,
    const int32_t* __restrict__ rank4,
    float* __restrict__ outputs,
    int input_channels,
    int selected_tokens,
    int base_tokens,
    int tokens2,
    int tokens3,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int loop) {
  const int batch_head = blockIdx.x;
  const int selected = blockIdx.y;
  const int logical = indices[batch_head * selected_tokens + selected];
  if (logical < 0 || logical >= base_tokens) {
    if (threadIdx.x == 0) {
      outputs[batch_head * selected_tokens + selected] = 0.0f;
    }
    return;
  }
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  float accumulated = 0.0f;
#pragma unroll
  for (int stream = 0; stream < 4; ++stream) {
    if (stream > loop) {
      continue;
    }
    int physical = logical;
    if (stream == 2) {
      physical = rank3[logical];
    } else if (stream == 3) {
      physical = rank4[logical];
    }
    if (physical < 0) {
      continue;
    }
    const int stream_tokens = stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
    const int packed_tokens = (stream_tokens + kPackFactor - 1) / kPackFactor;
    const int token_groups = (stream_tokens + group_size - 1) / group_size;
    const uint32_t* weights = stream == 0 ? weights0 :
        (stream == 1 ? weights1 : (stream == 2 ? weights2 : weights3));
    const half* scales = stream == 0 ? scales0 :
        (stream == 1 ? scales1 : (stream == 2 ? scales2 : scales3));
    const half* zeros = stream == 0 ? zeros0 :
        (stream == 1 ? zeros1 : (stream == 2 ? zeros2 : zeros3));
    const int packed_row = physical / kPackFactor;
    const int shift = (physical % kPackFactor) * 4;
    const int group = physical / group_size;
    float partial = 0.0f;
    for (int channel = threadIdx.x; channel < input_channels; channel += kWarpSize) {
      const uint32_t word =
          weights[(kv_batch_head * packed_tokens + packed_row) * input_channels + channel];
      const float code = static_cast<float>((word >> shift) & 0x0F);
      const float scale = __half2float(
          scales[(kv_batch_head * token_groups + group) * input_channels + channel]);
      const float zero = __half2float(
          zeros[(kv_batch_head * token_groups + group) * input_channels + channel]);
      partial += __half2float(inputs[batch_head * input_channels + channel]) *
                 (scale * code + zero);
    }
    const float reduced = warp_reduce_sum(partial);
    if (threadIdx.x == 0) {
      accumulated += __half2float(__float2half_rn(reduced));
    }
  }
  if (threadIdx.x == 0) {
    outputs[batch_head * selected_tokens + selected] = accumulated;
  }
}

__global__ void kivi_pv_cross_loop4_kernel(
    const half* __restrict__ inputs,
    const uint32_t* __restrict__ weights0,
    const uint32_t* __restrict__ weights1,
    const uint32_t* __restrict__ weights2,
    const uint32_t* __restrict__ weights3,
    const half* __restrict__ scales0,
    const half* __restrict__ scales1,
    const half* __restrict__ scales2,
    const half* __restrict__ scales3,
    const half* __restrict__ zeros0,
    const half* __restrict__ zeros1,
    const half* __restrict__ zeros2,
    const half* __restrict__ zeros3,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ rank3,
    const int32_t* __restrict__ rank4,
    float* __restrict__ outputs,
    int selected_tokens,
    int base_tokens,
    int tokens2,
    int tokens3,
    int output_channels,
    int packed_output_channels,
    int output_groups,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int loop) {
  const int batch_head = blockIdx.x;
  const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (packed_oc_idx >= packed_output_channels) {
    return;
  }
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const int oc_start = packed_oc_idx * kPackFactor;
  const int group = oc_start / group_size;
  float accumulated[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int stream = 0; stream < 4; ++stream) {
    if (stream > loop) {
      continue;
    }
    const int stream_tokens = stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
    const uint32_t* weights = stream == 0 ? weights0 :
        (stream == 1 ? weights1 : (stream == 2 ? weights2 : weights3));
    const half* scales = stream == 0 ? scales0 :
        (stream == 1 ? scales1 : (stream == 2 ? scales2 : scales3));
    const half* zeros = stream == 0 ? zeros0 :
        (stream == 1 ? zeros1 : (stream == 2 ? zeros2 : zeros3));
    float partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};
    for (int selected = threadIdx.x; selected < selected_tokens; selected += kWarpSize) {
      const int logical = indices[batch_head * selected_tokens + selected];
      if (logical < 0 || logical >= base_tokens) {
        continue;
      }
      int physical = logical;
      if (stream == 2) {
        physical = rank3[logical];
      } else if (stream == 3) {
        physical = rank4[logical];
      }
      if (physical < 0) {
        continue;
      }
      uint32_t word =
          weights[(kv_batch_head * packed_output_channels + packed_oc_idx) *
                      stream_tokens + physical];
      const float input = __half2float(inputs[batch_head * selected_tokens + selected]);
      const float scale = __half2float(
          scales[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
      const float zero = __half2float(
          zeros[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const float code = static_cast<float>(word & 0x0F);
        partial[packed_index] += input * (scale * code + zero);
        word >>= 4;
      }
    }
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const float reduced = warp_reduce_sum(partial[packed_index]);
      if (threadIdx.x == 0) {
        accumulated[packed_index] += __half2float(__float2half_rn(reduced));
      }
    }
  }
#pragma unroll
  for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
    const int output_channel = oc_start + packed_index;
    if (threadIdx.x == 0 && output_channel < output_channels) {
      outputs[batch_head * output_channels + output_channel] = accumulated[packed_index];
    }
  }
}

__global__ void kivi_sparse_attention_delta4_kernel(
    const half* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_bf16,
    const __nv_bfloat16* __restrict__ current_key,
    const __nv_bfloat16* __restrict__ current_value,
    const uint32_t* __restrict__ key_weights0,
    const uint32_t* __restrict__ key_weights1,
    const uint32_t* __restrict__ key_weights2,
    const uint32_t* __restrict__ key_weights3,
    const half* __restrict__ key_scales0,
    const half* __restrict__ key_scales1,
    const half* __restrict__ key_scales2,
    const half* __restrict__ key_scales3,
    const half* __restrict__ key_zeros0,
    const half* __restrict__ key_zeros1,
    const half* __restrict__ key_zeros2,
    const half* __restrict__ key_zeros3,
    const uint32_t* __restrict__ value_weights0,
    const uint32_t* __restrict__ value_weights1,
    const uint32_t* __restrict__ value_weights2,
    const uint32_t* __restrict__ value_weights3,
    const half* __restrict__ value_scales0,
    const half* __restrict__ value_scales1,
    const half* __restrict__ value_scales2,
    const half* __restrict__ value_scales3,
    const half* __restrict__ value_zeros0,
    const half* __restrict__ value_zeros1,
    const half* __restrict__ value_zeros2,
    const half* __restrict__ value_zeros3,
    const int32_t* __restrict__ indices,
    const bool* __restrict__ valid,
    const int32_t* __restrict__ rank3,
    const int32_t* __restrict__ rank4,
    const __nv_bfloat16* __restrict__ tail_key,
    const __nv_bfloat16* __restrict__ tail_value,
    const float* __restrict__ source_output,
    const float* __restrict__ global_mass,
    float* __restrict__ outputs,
    float* __restrict__ current_logit_outputs,
    float* __restrict__ full_logit_outputs,
    int selected_tokens,
    int base_tokens,
    int tokens2,
    int tokens3,
    int tail_tokens,
    int channels,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int current_position,
    float scaling,
    int loop,
    bool dense_source_mode) {
  const int batch_head = blockIdx.x;
  const int thread = threadIdx.x;
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;
  constexpr int kWarps = 4;
  extern __shared__ float shared[];
  float* attention = shared;
  float* reduction = shared + selected_tokens;
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;

  for (int selected = warp; selected < selected_tokens; selected += kWarps) {
    const int offset = batch_head * selected_tokens + selected;
    const bool is_valid = dense_source_mode || valid[offset];
    if (!is_valid) {
      if (lane == 0) {
        attention[selected] = -1.0e20f;
      }
      continue;
    }
    const int logical = dense_source_mode ? selected : indices[offset];
    float logit = 0.0f;
    if (logical == current_position) {
      for (int channel = lane; channel < channels; channel += kWarpSize) {
        logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
                 __bfloat162float(current_key[batch_head * channels + channel]);
      }
      logit = warp_reduce_sum(logit);
    } else if (logical >= base_tokens) {
      const int tail = logical - base_tokens;
      if (tail >= 0 && tail < tail_tokens) {
        for (int channel = lane; channel < channels; channel += kWarpSize) {
          logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
                   __bfloat162float(
                       tail_key[(batch_head * tail_tokens + tail) * channels + channel]);
        }
        logit = warp_reduce_sum(logit);
      } else {
        logit = lane == 0 ? -1.0e20f : 0.0f;
      }
    } else if (logical >= 0) {
      float accumulated = 0.0f;
#pragma unroll
      for (int stream = 0; stream < 4; ++stream) {
        if (stream > loop) {
          continue;
        }
        int physical = logical;
        if (stream == 2) {
          physical = rank3[logical];
        } else if (stream == 3) {
          physical = rank4[logical];
        }
        if (physical < 0) {
          continue;
        }
        const int stream_tokens =
            stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
        const int packed_tokens = (stream_tokens + kPackFactor - 1) / kPackFactor;
        const int token_groups = (stream_tokens + group_size - 1) / group_size;
        const uint32_t* weights = stream == 0 ? key_weights0 :
            (stream == 1 ? key_weights1 : (stream == 2 ? key_weights2 : key_weights3));
        const half* scales = stream == 0 ? key_scales0 :
            (stream == 1 ? key_scales1 : (stream == 2 ? key_scales2 : key_scales3));
        const half* zeros = stream == 0 ? key_zeros0 :
            (stream == 1 ? key_zeros1 : (stream == 2 ? key_zeros2 : key_zeros3));
        const int packed_row = physical / kPackFactor;
        const int shift = (physical % kPackFactor) * 4;
        const int group = physical / group_size;
        float partial = 0.0f;
        for (int channel = lane; channel < channels; channel += kWarpSize) {
          const uint32_t word =
              weights[(kv_batch_head * packed_tokens + packed_row) * channels + channel];
          const float code = static_cast<float>((word >> shift) & 0x0F);
          const float scale = __half2float(
              scales[(kv_batch_head * token_groups + group) * channels + channel]);
          const float zero = __half2float(
              zeros[(kv_batch_head * token_groups + group) * channels + channel]);
          partial += __half2float(query[batch_head * channels + channel]) *
                     (scale * code + zero);
        }
        const float reduced = warp_reduce_sum(partial);
        if (lane == 0) {
          accumulated += __half2float(__float2half_rn(reduced));
        }
      }
      logit = accumulated;
    } else {
      logit = lane == 0 ? -1.0e20f : 0.0f;
    }
    if (lane == 0) {
      attention[selected] = logit * scaling;
      if (full_logit_outputs != nullptr) {
        full_logit_outputs[offset] = attention[selected];
      }
      if (current_logit_outputs != nullptr && logical == current_position) {
        current_logit_outputs[batch_head] = attention[selected];
      }
    }
  }
  __syncthreads();

  float local_max = -1.0e20f;
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    local_max = fmaxf(local_max, attention[selected]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane == 0) {
    reduction[warp] = local_max;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarps ? reduction[lane] : -1.0e20f;
    value = warp_reduce_max(value);
    if (lane == 0) {
      reduction[0] = value;
    }
  }
  __syncthreads();
  const float maximum = reduction[0];
  float local_sum = 0.0f;
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    const float value = expf(attention[selected] - maximum);
    attention[selected] = value;
    local_sum += value;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    reduction[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarps ? reduction[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) {
      reduction[0] = value;
    }
  }
  __syncthreads();
  const float normalizer = reduction[0];

  const int packed_output_channels = (channels + kPackFactor - 1) / kPackFactor;
  const int output_groups = (channels + group_size - 1) / group_size;
  for (int packed_channel = warp; packed_channel < packed_output_channels;
       packed_channel += kWarps) {
    const int channel_start = packed_channel * kPackFactor;
    const int group = channel_start / group_size;
    float target[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f};

    float dense_partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
    for (int selected = lane; selected < selected_tokens; selected += kWarpSize) {
      const int offset = batch_head * selected_tokens + selected;
      const bool is_valid = dense_source_mode || valid[offset];
      if (!is_valid) {
        continue;
      }
      const float probability = attention[selected] / normalizer;
      const int logical = dense_source_mode ? selected : indices[offset];
      if (logical == current_position) {
#pragma unroll
        for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
          const int channel = channel_start + packed_index;
          if (channel < channels) {
            dense_partial[packed_index] += probability * __bfloat162float(
                current_value[batch_head * channels + channel]);
          }
        }
      } else if (logical >= base_tokens) {
        const int tail = logical - base_tokens;
        if (tail >= 0 && tail < tail_tokens) {
#pragma unroll
          for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
            const int channel = channel_start + packed_index;
            if (channel < channels) {
              dense_partial[packed_index] += probability * __bfloat162float(
                  tail_value[(batch_head * tail_tokens + tail) * channels + channel]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const float reduced = warp_reduce_sum(dense_partial[packed_index]);
      if (lane == 0) {
        target[packed_index] = reduced;
      }
    }

#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      if (stream > loop) {
        continue;
      }
      const int stream_tokens =
          stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
      const uint32_t* weights = stream == 0 ? value_weights0 :
          (stream == 1 ? value_weights1 :
           (stream == 2 ? value_weights2 : value_weights3));
      const half* scales = stream == 0 ? value_scales0 :
          (stream == 1 ? value_scales1 :
           (stream == 2 ? value_scales2 : value_scales3));
      const half* zeros = stream == 0 ? value_zeros0 :
          (stream == 1 ? value_zeros1 :
           (stream == 2 ? value_zeros2 : value_zeros3));
      float partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f};
      for (int selected = lane; selected < selected_tokens; selected += kWarpSize) {
        const int offset = batch_head * selected_tokens + selected;
        const bool is_valid = dense_source_mode || valid[offset];
        if (!is_valid) {
          continue;
        }
        const int logical = dense_source_mode ? selected : indices[offset];
        if (logical < 0 || logical >= base_tokens) {
          continue;
        }
        int physical = logical;
        if (stream == 2) {
          physical = rank3[logical];
        } else if (stream == 3) {
          physical = rank4[logical];
        }
        if (physical < 0) {
          continue;
        }
        uint32_t word =
            weights[(kv_batch_head * packed_output_channels + packed_channel) *
                        stream_tokens + physical];
        const float probability = __half2float(__float2half_rn(
            attention[selected] / normalizer));
        const float scale = __half2float(
            scales[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
        const float zero = __half2float(
            zeros[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
#pragma unroll
        for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
          const float code = static_cast<float>(word & 0x0F);
          partial[packed_index] += probability * (scale * code + zero);
          word >>= 4;
        }
      }
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const float reduced = warp_reduce_sum(partial[packed_index]);
        if (lane == 0) {
          target[packed_index] += __half2float(__float2half_rn(reduced));
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const int channel = channel_start + packed_index;
        if (channel < channels) {
          outputs[batch_head * channels + channel] = dense_source_mode
              ? target[packed_index]
              : global_mass[batch_head] *
                    (target[packed_index] - source_output[batch_head * channels + channel]);
        }
      }
    }
  }
}

// Long-context dense-source attention needs substantially more parallelism than
// one block per head.  The three kernels below keep the public fused extension
// entry point, but tile QK over tokens and PV over output channels.  This is
// numerically equivalent to the single-block reduction; only the reduction
// order changes.
__global__ void kivi_attention_logits_delta4_tiled_kernel(
    const half* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_bf16,
    const __nv_bfloat16* __restrict__ current_key,
    const uint32_t* __restrict__ key_weights0,
    const uint32_t* __restrict__ key_weights1,
    const half* __restrict__ key_scales0,
    const half* __restrict__ key_scales1,
    const half* __restrict__ key_zeros0,
    const half* __restrict__ key_zeros1,
    const __nv_bfloat16* __restrict__ tail_key,
    float* __restrict__ logits,
    int selected_tokens,
    int base_tokens,
    int tail_tokens,
    int channels,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int current_position,
    float scaling,
    int loop) {
  constexpr int kWarps = 4;
  const int batch_head = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int logical = blockIdx.y * kWarps + warp;
  if (logical >= selected_tokens) {
    return;
  }

  float logit = 0.0f;
  if (logical == current_position) {
    for (int channel = lane; channel < channels; channel += kWarpSize) {
      logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
               __bfloat162float(current_key[batch_head * channels + channel]);
    }
    logit = warp_reduce_sum(logit);
  } else if (logical >= base_tokens) {
    const int tail = logical - base_tokens;
    if (tail >= 0 && tail < tail_tokens) {
      for (int channel = lane; channel < channels; channel += kWarpSize) {
        logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
                 __bfloat162float(
                     tail_key[(batch_head * tail_tokens + tail) * channels + channel]);
      }
      logit = warp_reduce_sum(logit);
    } else {
      logit = lane == 0 ? -1.0e20f : 0.0f;
    }
  } else {
    const int head_ratio = num_heads / num_kv_heads;
    const int kv_batch_head = batch_head / head_ratio;
    const int packed_tokens = (base_tokens + kPackFactor - 1) / kPackFactor;
    const int token_groups = (base_tokens + group_size - 1) / group_size;
    const int packed_row = logical / kPackFactor;
    const int shift = (logical % kPackFactor) * 4;
    const int group = logical / group_size;
    float accumulated = 0.0f;
#pragma unroll
    for (int stream = 0; stream < 2; ++stream) {
      if (stream > loop) {
        continue;
      }
      const uint32_t* weights = stream == 0 ? key_weights0 : key_weights1;
      const half* scales = stream == 0 ? key_scales0 : key_scales1;
      const half* zeros = stream == 0 ? key_zeros0 : key_zeros1;
      float partial = 0.0f;
      for (int channel = lane; channel < channels; channel += kWarpSize) {
        const uint32_t word =
            weights[(kv_batch_head * packed_tokens + packed_row) * channels + channel];
        const float code = static_cast<float>((word >> shift) & 0x0F);
        const float scale = __half2float(
            scales[(kv_batch_head * token_groups + group) * channels + channel]);
        const float zero = __half2float(
            zeros[(kv_batch_head * token_groups + group) * channels + channel]);
        partial += __half2float(query[batch_head * channels + channel]) *
                   (scale * code + zero);
      }
      const float reduced = warp_reduce_sum(partial);
      if (lane == 0) {
        accumulated += __half2float(__float2half_rn(reduced));
      }
    }
    logit = accumulated;
  }
  if (lane == 0) {
    logits[batch_head * selected_tokens + logical] = logit * scaling;
  }
}

__global__ void kivi_attention_softmax_rows_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probabilities,
    int selected_tokens) {
  constexpr int kWarps = 8;
  __shared__ float reduction[kWarps];
  const int batch_head = blockIdx.x;
  const int thread = threadIdx.x;
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;
  const float* row = logits + batch_head * selected_tokens;
  float* output = probabilities + batch_head * selected_tokens;

  float local_max = -1.0e20f;
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    local_max = fmaxf(local_max, row[selected]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane == 0) {
    reduction[warp] = local_max;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarps ? reduction[lane] : -1.0e20f;
    value = warp_reduce_max(value);
    if (lane == 0) {
      reduction[0] = value;
    }
  }
  __syncthreads();
  const float maximum = reduction[0];

  float local_sum = 0.0f;
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    const float value = expf(row[selected] - maximum);
    output[selected] = value;
    local_sum += value;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    reduction[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarps ? reduction[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) {
      reduction[0] = value;
    }
  }
  __syncthreads();
  const float normalizer = reduction[0];
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    output[selected] /= normalizer;
  }
}

__global__ void kivi_attention_pv_delta4_tiled_kernel(
    const float* __restrict__ probabilities,
    const __nv_bfloat16* __restrict__ current_value,
    const uint32_t* __restrict__ value_weights0,
    const uint32_t* __restrict__ value_weights1,
    const half* __restrict__ value_scales0,
    const half* __restrict__ value_scales1,
    const half* __restrict__ value_zeros0,
    const half* __restrict__ value_zeros1,
    const __nv_bfloat16* __restrict__ tail_value,
    float* __restrict__ outputs,
    int selected_tokens,
    int base_tokens,
    int tail_tokens,
    int channels,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int current_position,
    int loop) {
  constexpr int kWarps = 8;
  __shared__ float partials[kWarps * kPackFactor];
  const int batch_head = blockIdx.x;
  const int packed_channel = blockIdx.y;
  const int thread = threadIdx.x;
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;
  const int channel_start = packed_channel * kPackFactor;
  const int output_groups = (channels + group_size - 1) / group_size;
  const int group = channel_start / group_size;
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const float* attention = probabilities + batch_head * selected_tokens;
  float target[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};

  float dense_partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 0.0f};
  for (int logical = thread; logical < selected_tokens; logical += blockDim.x) {
    const float probability = attention[logical];
    if (logical == current_position) {
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const int channel = channel_start + packed_index;
        if (channel < channels) {
          dense_partial[packed_index] += probability * __bfloat162float(
              current_value[batch_head * channels + channel]);
        }
      }
    } else if (logical >= base_tokens) {
      const int tail = logical - base_tokens;
      if (tail >= 0 && tail < tail_tokens) {
#pragma unroll
        for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
          const int channel = channel_start + packed_index;
          if (channel < channels) {
            dense_partial[packed_index] += probability * __bfloat162float(
                tail_value[(batch_head * tail_tokens + tail) * channels + channel]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
    const float reduced = warp_reduce_sum(dense_partial[packed_index]);
    if (lane == 0) {
      partials[warp * kPackFactor + packed_index] = reduced;
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      float value = lane < kWarps
          ? partials[lane * kPackFactor + packed_index]
          : 0.0f;
      value = warp_reduce_sum(value);
      if (lane == 0) {
        target[packed_index] = value;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int stream = 0; stream < 2; ++stream) {
    if (stream > loop) {
      continue;
    }
    const uint32_t* weights = stream == 0 ? value_weights0 : value_weights1;
    const half* scales = stream == 0 ? value_scales0 : value_scales1;
    const half* zeros = stream == 0 ? value_zeros0 : value_zeros1;
    float packed_partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    for (int logical = thread; logical < base_tokens; logical += blockDim.x) {
      uint32_t word =
          weights[(kv_batch_head * ((channels + kPackFactor - 1) / kPackFactor) +
                   packed_channel) * base_tokens + logical];
      const float probability = __half2float(__float2half_rn(attention[logical]));
      const float scale = __half2float(
          scales[(kv_batch_head * output_groups + group) * base_tokens + logical]);
      const float zero = __half2float(
          zeros[(kv_batch_head * output_groups + group) * base_tokens + logical]);
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const float code = static_cast<float>(word & 0x0F);
        packed_partial[packed_index] += probability * (scale * code + zero);
        word >>= 4;
      }
    }
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const float reduced = warp_reduce_sum(packed_partial[packed_index]);
      if (lane == 0) {
        partials[warp * kPackFactor + packed_index] = reduced;
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        float value = lane < kWarps
            ? partials[lane * kPackFactor + packed_index]
            : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
          target[packed_index] += __half2float(__float2half_rn(value));
        }
      }
    }
    __syncthreads();
  }

  if (thread == 0) {
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const int channel = channel_start + packed_index;
      if (channel < channels) {
        outputs[batch_head * channels + channel] = target[packed_index];
      }
    }
  }
}

__global__ void kivi_sparse_logits_delta4_tiled_kernel(
    const half* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_bf16,
    const __nv_bfloat16* __restrict__ current_key,
    const uint32_t* __restrict__ key_weights0,
    const uint32_t* __restrict__ key_weights1,
    const uint32_t* __restrict__ key_weights2,
    const uint32_t* __restrict__ key_weights3,
    const half* __restrict__ key_scales0,
    const half* __restrict__ key_scales1,
    const half* __restrict__ key_scales2,
    const half* __restrict__ key_scales3,
    const half* __restrict__ key_zeros0,
    const half* __restrict__ key_zeros1,
    const half* __restrict__ key_zeros2,
    const half* __restrict__ key_zeros3,
    const int32_t* __restrict__ indices,
    const bool* __restrict__ valid,
    const int32_t* __restrict__ rank3,
    const int32_t* __restrict__ rank4,
    const __nv_bfloat16* __restrict__ tail_key,
    float* __restrict__ logits,
    float* __restrict__ current_logits,
    int selected_tokens,
    int base_tokens,
    int tokens2,
    int tokens3,
    int tail_tokens,
    int channels,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int current_position,
    float scaling,
    int loop) {
  constexpr int kWarps = 4;
  const int batch_head = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int selected = blockIdx.y * kWarps + warp;
  if (selected >= selected_tokens) {
    return;
  }
  const int offset = batch_head * selected_tokens + selected;
  if (!valid[offset]) {
    if (lane == 0) {
      logits[offset] = -1.0e20f;
    }
    return;
  }
  const int logical = indices[offset];
  float logit = 0.0f;
  if (logical == current_position) {
    for (int channel = lane; channel < channels; channel += kWarpSize) {
      logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
               __bfloat162float(current_key[batch_head * channels + channel]);
    }
    logit = warp_reduce_sum(logit);
  } else if (logical >= base_tokens) {
    const int tail = logical - base_tokens;
    if (tail >= 0 && tail < tail_tokens) {
      for (int channel = lane; channel < channels; channel += kWarpSize) {
        logit += __bfloat162float(query_bf16[batch_head * channels + channel]) *
                 __bfloat162float(
                     tail_key[(batch_head * tail_tokens + tail) * channels + channel]);
      }
      logit = warp_reduce_sum(logit);
    } else {
      logit = lane == 0 ? -1.0e20f : 0.0f;
    }
  } else if (logical >= 0) {
    const int head_ratio = num_heads / num_kv_heads;
    const int kv_batch_head = batch_head / head_ratio;
    float accumulated = 0.0f;
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      if (stream > loop) {
        continue;
      }
      int physical = logical;
      if (stream == 2) {
        physical = rank3[logical];
      } else if (stream == 3) {
        physical = rank4[logical];
      }
      if (physical < 0) {
        continue;
      }
      const int stream_tokens =
          stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
      const int packed_tokens = (stream_tokens + kPackFactor - 1) / kPackFactor;
      const int token_groups = (stream_tokens + group_size - 1) / group_size;
      const uint32_t* weights = stream == 0 ? key_weights0 :
          (stream == 1 ? key_weights1 : (stream == 2 ? key_weights2 : key_weights3));
      const half* scales = stream == 0 ? key_scales0 :
          (stream == 1 ? key_scales1 : (stream == 2 ? key_scales2 : key_scales3));
      const half* zeros = stream == 0 ? key_zeros0 :
          (stream == 1 ? key_zeros1 : (stream == 2 ? key_zeros2 : key_zeros3));
      const int packed_row = physical / kPackFactor;
      const int shift = (physical % kPackFactor) * 4;
      const int group = physical / group_size;
      float partial = 0.0f;
      for (int channel = lane; channel < channels; channel += kWarpSize) {
        const uint32_t word =
            weights[(kv_batch_head * packed_tokens + packed_row) * channels + channel];
        const float code = static_cast<float>((word >> shift) & 0x0F);
        const float scale = __half2float(
            scales[(kv_batch_head * token_groups + group) * channels + channel]);
        const float zero = __half2float(
            zeros[(kv_batch_head * token_groups + group) * channels + channel]);
        partial += __half2float(query[batch_head * channels + channel]) *
                   (scale * code + zero);
      }
      const float reduced = warp_reduce_sum(partial);
      if (lane == 0) {
        accumulated += __half2float(__float2half_rn(reduced));
      }
    }
    logit = accumulated;
  } else {
    logit = lane == 0 ? -1.0e20f : 0.0f;
  }
  if (lane == 0) {
    const float scaled = logit * scaling;
    logits[offset] = scaled;
    if (current_logits != nullptr && logical == current_position) {
      current_logits[batch_head] = scaled;
    }
  }
}

__global__ void kivi_sparse_pv_delta4_tiled_kernel(
    const float* __restrict__ probabilities,
    const __nv_bfloat16* __restrict__ current_value,
    const uint32_t* __restrict__ value_weights0,
    const uint32_t* __restrict__ value_weights1,
    const uint32_t* __restrict__ value_weights2,
    const uint32_t* __restrict__ value_weights3,
    const half* __restrict__ value_scales0,
    const half* __restrict__ value_scales1,
    const half* __restrict__ value_scales2,
    const half* __restrict__ value_scales3,
    const half* __restrict__ value_zeros0,
    const half* __restrict__ value_zeros1,
    const half* __restrict__ value_zeros2,
    const half* __restrict__ value_zeros3,
    const int32_t* __restrict__ indices,
    const bool* __restrict__ valid,
    const int32_t* __restrict__ rank3,
    const int32_t* __restrict__ rank4,
    const __nv_bfloat16* __restrict__ tail_value,
    const float* __restrict__ source_output,
    const float* __restrict__ global_mass,
    float* __restrict__ outputs,
    int selected_tokens,
    int base_tokens,
    int tokens2,
    int tokens3,
    int tail_tokens,
    int channels,
    int group_size,
    int num_heads,
    int num_kv_heads,
    int current_position,
    int loop) {
  constexpr int kWarps = 8;
  __shared__ float partials[kWarps * kPackFactor];
  const int batch_head = blockIdx.x;
  const int packed_channel = blockIdx.y;
  const int thread = threadIdx.x;
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;
  const int channel_start = packed_channel * kPackFactor;
  const int output_groups = (channels + group_size - 1) / group_size;
  const int group = channel_start / group_size;
  const int head_ratio = num_heads / num_kv_heads;
  const int kv_batch_head = batch_head / head_ratio;
  const float* attention = probabilities + batch_head * selected_tokens;
  float target[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};

  float dense_partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 0.0f};
  for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
    const int offset = batch_head * selected_tokens + selected;
    if (!valid[offset]) {
      continue;
    }
    const int logical = indices[offset];
    const float probability = attention[selected];
    if (logical == current_position) {
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const int channel = channel_start + packed_index;
        if (channel < channels) {
          dense_partial[packed_index] += probability * __bfloat162float(
              current_value[batch_head * channels + channel]);
        }
      }
    } else if (logical >= base_tokens) {
      const int tail = logical - base_tokens;
      if (tail >= 0 && tail < tail_tokens) {
#pragma unroll
        for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
          const int channel = channel_start + packed_index;
          if (channel < channels) {
            dense_partial[packed_index] += probability * __bfloat162float(
                tail_value[(batch_head * tail_tokens + tail) * channels + channel]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
    const float reduced = warp_reduce_sum(dense_partial[packed_index]);
    if (lane == 0) {
      partials[warp * kPackFactor + packed_index] = reduced;
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      float value = lane < kWarps
          ? partials[lane * kPackFactor + packed_index]
          : 0.0f;
      value = warp_reduce_sum(value);
      if (lane == 0) {
        target[packed_index] = value;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int stream = 0; stream < 4; ++stream) {
    if (stream > loop) {
      continue;
    }
    const int stream_tokens =
        stream < 2 ? base_tokens : (stream == 2 ? tokens2 : tokens3);
    const uint32_t* weights = stream == 0 ? value_weights0 :
        (stream == 1 ? value_weights1 : (stream == 2 ? value_weights2 : value_weights3));
    const half* scales = stream == 0 ? value_scales0 :
        (stream == 1 ? value_scales1 : (stream == 2 ? value_scales2 : value_scales3));
    const half* zeros = stream == 0 ? value_zeros0 :
        (stream == 1 ? value_zeros1 : (stream == 2 ? value_zeros2 : value_zeros3));
    float packed_partial[kPackFactor] = {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    for (int selected = thread; selected < selected_tokens; selected += blockDim.x) {
      const int offset = batch_head * selected_tokens + selected;
      if (!valid[offset]) {
        continue;
      }
      const int logical = indices[offset];
      if (logical < 0 || logical >= base_tokens) {
        continue;
      }
      int physical = logical;
      if (stream == 2) {
        physical = rank3[logical];
      } else if (stream == 3) {
        physical = rank4[logical];
      }
      if (physical < 0) {
        continue;
      }
      uint32_t word =
          weights[(kv_batch_head * ((channels + kPackFactor - 1) / kPackFactor) +
                   packed_channel) * stream_tokens + physical];
      const float probability = __half2float(__float2half_rn(attention[selected]));
      const float scale = __half2float(
          scales[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
      const float zero = __half2float(
          zeros[(kv_batch_head * output_groups + group) * stream_tokens + physical]);
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        const float code = static_cast<float>(word & 0x0F);
        packed_partial[packed_index] += probability * (scale * code + zero);
        word >>= 4;
      }
    }
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const float reduced = warp_reduce_sum(packed_partial[packed_index]);
      if (lane == 0) {
        partials[warp * kPackFactor + packed_index] = reduced;
      }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
        float value = lane < kWarps
            ? partials[lane * kPackFactor + packed_index]
            : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
          target[packed_index] += __half2float(__float2half_rn(value));
        }
      }
    }
    __syncthreads();
  }

  if (thread == 0) {
#pragma unroll
    for (int packed_index = 0; packed_index < kPackFactor; ++packed_index) {
      const int channel = channel_start + packed_index;
      if (channel < channels) {
        outputs[batch_head * channels + channel] = global_mass[batch_head] *
            (target[packed_index] - source_output[batch_head * channels + channel]);
      }
    }
  }
}

}  // namespace

torch::Tensor gemv_forward_cuda_outer_dim_logical(
    torch::Tensor inputs,
    torch::Tensor kernel,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    int64_t bit,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t output_channels) {
  TORCH_CHECK(inputs.is_cuda() && kernel.is_cuda() && scaling_factors.is_cuda() &&
                  zeros.is_cuda(),
              "all KIVI GEMV tensors must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half,
              "KIVI GEMV inputs must be float16");
  TORCH_CHECK(kernel.scalar_type() == at::ScalarType::Int,
              "KIVI GEMV payload must be int32");
  TORCH_CHECK(scaling_factors.scalar_type() == at::ScalarType::Half &&
                  zeros.scalar_type() == at::ScalarType::Half,
              "KIVI GEMV metadata must be float16");
  TORCH_CHECK(inputs.is_contiguous() && kernel.is_contiguous() &&
                  scaling_factors.is_contiguous() && zeros.is_contiguous(),
              "KIVI GEMV tensors must be contiguous");
  TORCH_CHECK(inputs.dim() == 3 && kernel.dim() == 3 &&
                  scaling_factors.dim() == 3 && zeros.dim() == 3,
              "KIVI GEMV tensors must be rank three");
  TORCH_CHECK(bit == 4, "FlashLoop currently specializes 4-bit KIVI GEMV");
  TORCH_CHECK(group_size == 64,
              "FlashLoop currently specializes KIVI group_size=64");
  TORCH_CHECK(num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
              "invalid query/KV head ratio");
  TORCH_CHECK(output_channels > 0, "logical output width must be positive");

  const int64_t batch_heads = inputs.size(0);
  const int64_t input_rows = inputs.size(1);
  const int64_t input_channels = inputs.size(2);
  const int64_t packed_output_channels = (output_channels + kPackFactor - 1) / kPackFactor;
  const int64_t output_groups = (output_channels + group_size - 1) / group_size;
  TORCH_CHECK(kernel.size(0) * num_heads == batch_heads * num_kv_heads,
              "packed KV batch/head count differs from inputs");
  TORCH_CHECK(kernel.size(1) == packed_output_channels &&
                  kernel.size(2) == input_channels,
              "packed payload shape differs from logical matrix");
  TORCH_CHECK(scaling_factors.sizes() == zeros.sizes() &&
                  scaling_factors.size(0) == kernel.size(0) &&
                  scaling_factors.size(1) == output_groups &&
                  scaling_factors.size(2) == input_channels,
              "affine metadata shape differs from logical matrix");

  auto output = torch::empty(
      {batch_heads, input_rows, output_channels}, inputs.options());
  const dim3 threads(kWarpSize, 4);
  const dim3 blocks(
      batch_heads,
      (packed_output_channels + threads.y - 1) / threads.y,
      input_rows);
  kivi_outer_gemv4_logical_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(kernel.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scaling_factors.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      static_cast<int>(input_rows),
      static_cast<int>(input_channels),
      static_cast<int>(output_channels),
      static_cast<int>(packed_output_channels),
      static_cast<int>(output_groups),
      static_cast<int>(group_size),
      static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qk_selected_cuda(
    torch::Tensor inputs,
    torch::Tensor kernel,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    torch::Tensor indices,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t logical_tokens) {
  TORCH_CHECK(inputs.is_cuda() && kernel.is_cuda() && scaling_factors.is_cuda() &&
                  zeros.is_cuda() && indices.is_cuda(),
              "all selected QK tensors must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half &&
                  kernel.scalar_type() == at::ScalarType::Int &&
                  scaling_factors.scalar_type() == at::ScalarType::Half &&
                  zeros.scalar_type() == at::ScalarType::Half &&
                  indices.scalar_type() == at::ScalarType::Int,
              "selected QK tensor dtypes differ from the CUDA ABI");
  TORCH_CHECK(inputs.is_contiguous() && kernel.is_contiguous() &&
                  scaling_factors.is_contiguous() && zeros.is_contiguous() &&
                  indices.is_contiguous(),
              "selected QK tensors must be contiguous");
  TORCH_CHECK(inputs.dim() == 3 && inputs.size(1) == 1 && indices.dim() == 2,
              "selected QK expects [BH,1,C] inputs and [BH,K] indices");
  TORCH_CHECK(group_size == 64 && logical_tokens > 0,
              "selected QK specializes group_size=64 and non-empty K");
  const int64_t batch_heads = inputs.size(0);
  const int64_t selected_tokens = indices.size(1);
  const int64_t input_channels = inputs.size(2);
  const int64_t packed_tokens = (logical_tokens + kPackFactor - 1) / kPackFactor;
  const int64_t token_groups = (logical_tokens + group_size - 1) / group_size;
  TORCH_CHECK(indices.size(0) == batch_heads &&
                  kernel.size(0) * num_heads == batch_heads * num_kv_heads &&
                  kernel.size(1) == packed_tokens &&
                  kernel.size(2) == input_channels,
              "selected QK payload shape differs from inputs");
  TORCH_CHECK(scaling_factors.sizes() == zeros.sizes() &&
                  scaling_factors.size(0) == kernel.size(0) &&
                  scaling_factors.size(1) == token_groups &&
                  scaling_factors.size(2) == input_channels,
              "selected QK metadata shape differs from K storage");
  auto output = torch::empty({batch_heads, 1, selected_tokens}, inputs.options());
  const dim3 blocks(batch_heads, selected_tokens);
  kivi_qk_selected4_kernel<<<blocks, kWarpSize, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(kernel.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scaling_factors.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      static_cast<int>(input_channels),
      static_cast<int>(selected_tokens),
      static_cast<int>(logical_tokens),
      static_cast<int>(packed_tokens),
      static_cast<int>(token_groups),
      static_cast<int>(group_size),
      static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor pv_selected_cuda(
    torch::Tensor inputs,
    torch::Tensor kernel,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    torch::Tensor indices,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t logical_tokens,
    int64_t output_channels) {
  TORCH_CHECK(inputs.is_cuda() && kernel.is_cuda() && scaling_factors.is_cuda() &&
                  zeros.is_cuda() && indices.is_cuda(),
              "all selected PV tensors must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half &&
                  kernel.scalar_type() == at::ScalarType::Int &&
                  scaling_factors.scalar_type() == at::ScalarType::Half &&
                  zeros.scalar_type() == at::ScalarType::Half &&
                  indices.scalar_type() == at::ScalarType::Int,
              "selected PV tensor dtypes differ from the CUDA ABI");
  TORCH_CHECK(inputs.is_contiguous() && kernel.is_contiguous() &&
                  scaling_factors.is_contiguous() && zeros.is_contiguous() &&
                  indices.is_contiguous(),
              "selected PV tensors must be contiguous");
  TORCH_CHECK(inputs.dim() == 3 && inputs.size(1) == 1 && indices.dim() == 2,
              "selected PV expects [BH,1,K] inputs and [BH,K] indices");
  TORCH_CHECK(group_size == 64 && logical_tokens > 0 && output_channels > 0,
              "selected PV requires valid logical matrix dimensions");
  const int64_t batch_heads = inputs.size(0);
  const int64_t selected_tokens = inputs.size(2);
  const int64_t packed_output = (output_channels + kPackFactor - 1) / kPackFactor;
  const int64_t output_groups = (output_channels + group_size - 1) / group_size;
  TORCH_CHECK(indices.size(0) == batch_heads &&
                  indices.size(1) == selected_tokens &&
                  kernel.size(0) * num_heads == batch_heads * num_kv_heads &&
                  kernel.size(1) == packed_output &&
                  kernel.size(2) == logical_tokens,
              "selected PV payload shape differs from inputs");
  TORCH_CHECK(scaling_factors.sizes() == zeros.sizes() &&
                  scaling_factors.size(0) == kernel.size(0) &&
                  scaling_factors.size(1) == output_groups &&
                  scaling_factors.size(2) == logical_tokens,
              "selected PV metadata shape differs from V storage");
  auto output = torch::empty({batch_heads, 1, output_channels}, inputs.options());
  const dim3 threads(kWarpSize, 4);
  const dim3 blocks(batch_heads, (packed_output + threads.y - 1) / threads.y);
  kivi_pv_selected4_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(kernel.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scaling_factors.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      static_cast<int>(selected_tokens),
      static_cast<int>(logical_tokens),
      static_cast<int>(output_channels),
      static_cast<int>(packed_output),
      static_cast<int>(output_groups),
      static_cast<int>(group_size),
      static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qk_cross_loop_cuda(
    torch::Tensor inputs,
    std::vector<torch::Tensor> kernels,
    std::vector<torch::Tensor> scaling_factors,
    std::vector<torch::Tensor> zeros,
    torch::Tensor indices,
    torch::Tensor rank3,
    torch::Tensor rank4,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t loop,
    std::vector<int64_t> logical_tokens) {
  TORCH_CHECK(kernels.size() == 4 && scaling_factors.size() == 4 &&
                  zeros.size() == 4 && logical_tokens.size() == 4,
              "fused QK requires four physical streams");
  TORCH_CHECK(inputs.is_cuda() && indices.is_cuda() && rank3.is_cuda() && rank4.is_cuda(),
              "fused QK control tensors must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half &&
                  indices.scalar_type() == at::ScalarType::Int &&
                  rank3.scalar_type() == at::ScalarType::Int &&
                  rank4.scalar_type() == at::ScalarType::Int,
              "fused QK control tensor dtypes differ from the CUDA ABI");
  TORCH_CHECK(inputs.is_contiguous() && indices.is_contiguous() &&
                  rank3.is_contiguous() && rank4.is_contiguous(),
              "fused QK control tensors must be contiguous");
  TORCH_CHECK(inputs.dim() == 3 && inputs.size(1) == 1 && indices.dim() == 2,
              "fused QK expects [BH,1,C] inputs and [BH,K] indices");
  TORCH_CHECK(group_size == 64 && loop >= 0 && loop < 4 &&
                  logical_tokens[0] > 0 && logical_tokens[1] == logical_tokens[0],
              "invalid fused QK configuration");
  const int64_t batch_heads = inputs.size(0);
  const int64_t selected_tokens = indices.size(1);
  const int64_t input_channels = inputs.size(2);
  TORCH_CHECK(indices.size(0) == batch_heads && rank3.numel() == logical_tokens[0] &&
                  rank4.numel() == logical_tokens[0],
              "fused QK logical maps differ from the anchor");
  for (int stream = 0; stream < 4; ++stream) {
    const int64_t tokens = logical_tokens[stream];
    const int64_t packed_tokens = (tokens + kPackFactor - 1) / kPackFactor;
    const int64_t token_groups = (tokens + group_size - 1) / group_size;
    TORCH_CHECK(kernels[stream].is_cuda() && scaling_factors[stream].is_cuda() &&
                    zeros[stream].is_cuda() &&
                    kernels[stream].scalar_type() == at::ScalarType::Int &&
                    scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    zeros[stream].scalar_type() == at::ScalarType::Half &&
                    kernels[stream].is_contiguous() &&
                    scaling_factors[stream].is_contiguous() && zeros[stream].is_contiguous(),
                "fused QK stream storage differs from the CUDA ABI");
    TORCH_CHECK(kernels[stream].size(0) * num_heads == batch_heads * num_kv_heads &&
                    kernels[stream].size(1) == packed_tokens &&
                    kernels[stream].size(2) == input_channels &&
                    scaling_factors[stream].sizes() == zeros[stream].sizes() &&
                    scaling_factors[stream].size(1) == token_groups &&
                    scaling_factors[stream].size(2) == input_channels,
                "fused QK stream shape differs from its logical matrix");
  }
  auto output = torch::empty(
      {batch_heads, 1, selected_tokens}, inputs.options().dtype(torch::kFloat32));
  const dim3 blocks(batch_heads, selected_tokens);
  kivi_qk_cross_loop4_kernel<<<blocks, kWarpSize, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[2].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[3].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[3].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[3].data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(), rank3.data_ptr<int32_t>(), rank4.data_ptr<int32_t>(),
      output.data_ptr<float>(), static_cast<int>(input_channels),
      static_cast<int>(selected_tokens), static_cast<int>(logical_tokens[0]),
      static_cast<int>(logical_tokens[2]), static_cast<int>(logical_tokens[3]),
      static_cast<int>(group_size), static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor pv_cross_loop_cuda(
    torch::Tensor inputs,
    std::vector<torch::Tensor> kernels,
    std::vector<torch::Tensor> scaling_factors,
    std::vector<torch::Tensor> zeros,
    torch::Tensor indices,
    torch::Tensor rank3,
    torch::Tensor rank4,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t loop,
    std::vector<int64_t> logical_tokens,
    int64_t output_channels) {
  TORCH_CHECK(kernels.size() == 4 && scaling_factors.size() == 4 &&
                  zeros.size() == 4 && logical_tokens.size() == 4,
              "fused PV requires four physical streams");
  TORCH_CHECK(inputs.is_cuda() && indices.is_cuda() && rank3.is_cuda() && rank4.is_cuda(),
              "fused PV control tensors must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half &&
                  indices.scalar_type() == at::ScalarType::Int &&
                  rank3.scalar_type() == at::ScalarType::Int &&
                  rank4.scalar_type() == at::ScalarType::Int,
              "fused PV control tensor dtypes differ from the CUDA ABI");
  TORCH_CHECK(inputs.is_contiguous() && indices.is_contiguous() &&
                  rank3.is_contiguous() && rank4.is_contiguous(),
              "fused PV control tensors must be contiguous");
  TORCH_CHECK(inputs.dim() == 3 && inputs.size(1) == 1 && indices.dim() == 2,
              "fused PV expects [BH,1,K] inputs and [BH,K] indices");
  TORCH_CHECK(group_size == 64 && loop >= 0 && loop < 4 && output_channels > 0 &&
                  logical_tokens[0] > 0 && logical_tokens[1] == logical_tokens[0],
              "invalid fused PV configuration");
  const int64_t batch_heads = inputs.size(0);
  const int64_t selected_tokens = inputs.size(2);
  const int64_t packed_output = (output_channels + kPackFactor - 1) / kPackFactor;
  const int64_t output_groups = (output_channels + group_size - 1) / group_size;
  TORCH_CHECK(indices.size(0) == batch_heads && indices.size(1) == selected_tokens &&
                  rank3.numel() == logical_tokens[0] && rank4.numel() == logical_tokens[0],
              "fused PV logical maps differ from the anchor");
  for (int stream = 0; stream < 4; ++stream) {
    const int64_t tokens = logical_tokens[stream];
    TORCH_CHECK(kernels[stream].is_cuda() && scaling_factors[stream].is_cuda() &&
                    zeros[stream].is_cuda() &&
                    kernels[stream].scalar_type() == at::ScalarType::Int &&
                    scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    zeros[stream].scalar_type() == at::ScalarType::Half &&
                    kernels[stream].is_contiguous() &&
                    scaling_factors[stream].is_contiguous() && zeros[stream].is_contiguous(),
                "fused PV stream storage differs from the CUDA ABI");
    TORCH_CHECK(kernels[stream].size(0) * num_heads == batch_heads * num_kv_heads &&
                    kernels[stream].size(1) == packed_output &&
                    kernels[stream].size(2) == tokens &&
                    scaling_factors[stream].sizes() == zeros[stream].sizes() &&
                    scaling_factors[stream].size(1) == output_groups &&
                    scaling_factors[stream].size(2) == tokens,
                "fused PV stream shape differs from its logical matrix");
  }
  auto output = torch::empty(
      {batch_heads, 1, output_channels}, inputs.options().dtype(torch::kFloat32));
  const dim3 threads(kWarpSize, 4);
  const dim3 blocks(batch_heads, (packed_output + threads.y - 1) / threads.y);
  kivi_pv_cross_loop4_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[2].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(kernels[3].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scaling_factors[3].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(zeros[3].data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(), rank3.data_ptr<int32_t>(), rank4.data_ptr<int32_t>(),
      output.data_ptr<float>(), static_cast<int>(selected_tokens),
      static_cast<int>(logical_tokens[0]), static_cast<int>(logical_tokens[2]),
      static_cast<int>(logical_tokens[3]), static_cast<int>(output_channels),
      static_cast<int>(packed_output), static_cast<int>(output_groups),
      static_cast<int>(group_size), static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> sparse_attention_delta_cuda(
    torch::Tensor query,
    torch::Tensor query_bf16,
    torch::Tensor current_key,
    torch::Tensor current_value,
    std::vector<torch::Tensor> key_kernels,
    std::vector<torch::Tensor> key_scaling_factors,
    std::vector<torch::Tensor> key_zeros,
    std::vector<torch::Tensor> value_kernels,
    std::vector<torch::Tensor> value_scaling_factors,
    std::vector<torch::Tensor> value_zeros,
    torch::Tensor indices,
    torch::Tensor valid,
    torch::Tensor rank3,
    torch::Tensor rank4,
    torch::Tensor tail_key,
    torch::Tensor tail_value,
    torch::Tensor source_output,
    torch::Tensor global_mass,
    int64_t current_position,
    double scaling,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t loop,
    std::vector<int64_t> logical_tokens,
    int64_t output_channels,
    bool return_current_logits) {
  TORCH_CHECK(key_kernels.size() == 4 && key_scaling_factors.size() == 4 &&
                  key_zeros.size() == 4 && value_kernels.size() == 4 &&
                  value_scaling_factors.size() == 4 && value_zeros.size() == 4 &&
                  logical_tokens.size() == 4,
              "fused sparse attention requires four K/V streams");
  TORCH_CHECK(query.is_cuda() && query_bf16.is_cuda() && current_key.is_cuda() && current_value.is_cuda() &&
                  indices.is_cuda() && valid.is_cuda() && rank3.is_cuda() && rank4.is_cuda() &&
                  tail_key.is_cuda() && tail_value.is_cuda() && source_output.is_cuda() &&
                  global_mass.is_cuda(),
              "fused sparse-attention tensors must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half &&
                  query_bf16.scalar_type() == at::ScalarType::BFloat16 &&
                  current_key.scalar_type() == at::ScalarType::BFloat16 &&
                  current_value.scalar_type() == at::ScalarType::BFloat16 &&
                  tail_key.scalar_type() == at::ScalarType::BFloat16 &&
                  tail_value.scalar_type() == at::ScalarType::BFloat16 &&
                  indices.scalar_type() == at::ScalarType::Int &&
                  valid.scalar_type() == at::ScalarType::Bool &&
                  rank3.scalar_type() == at::ScalarType::Int &&
                  rank4.scalar_type() == at::ScalarType::Int &&
                  source_output.scalar_type() == at::ScalarType::Float &&
                  global_mass.scalar_type() == at::ScalarType::Float,
              "fused sparse-attention dtypes differ from the CUDA ABI");
  TORCH_CHECK(query.is_contiguous() && query_bf16.is_contiguous() && current_key.is_contiguous() &&
                  current_value.is_contiguous() && indices.is_contiguous() &&
                  valid.is_contiguous() && rank3.is_contiguous() && rank4.is_contiguous() &&
                  tail_key.is_contiguous() && tail_value.is_contiguous() &&
                  source_output.is_contiguous() && global_mass.is_contiguous(),
              "fused sparse-attention tensors must be contiguous");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == 1 && query_bf16.dim() == 2 && indices.dim() == 2 &&
                  valid.sizes() == indices.sizes() && current_key.dim() == 2 &&
                  current_value.sizes() == current_key.sizes() && tail_key.dim() == 3 &&
                  tail_value.sizes() == tail_key.sizes() && source_output.dim() == 2 &&
                  global_mass.dim() == 1,
              "invalid fused sparse-attention tensor ranks");
  TORCH_CHECK(group_size == 64 && loop >= 2 && loop < 4 && output_channels > 0 &&
                  logical_tokens[0] > 0 && logical_tokens[1] == logical_tokens[0],
              "invalid fused sparse-attention configuration");
  const int64_t batch_heads = query.size(0);
  const int64_t selected_tokens = indices.size(1);
  const int64_t channels = query.size(2);
  const int64_t tail_tokens = tail_key.size(1);
  TORCH_CHECK(channels == output_channels && indices.size(0) == batch_heads &&
                  query_bf16.sizes() == current_key.sizes() &&
                  current_key.size(0) == batch_heads && current_key.size(1) == channels &&
                  tail_key.size(0) == batch_heads && tail_key.size(2) == channels &&
                  source_output.sizes() == current_key.sizes() &&
                  global_mass.numel() == batch_heads &&
                  rank3.numel() == logical_tokens[0] && rank4.numel() == logical_tokens[0],
              "fused sparse-attention shapes differ from the anchor");
  const int64_t packed_output = (channels + kPackFactor - 1) / kPackFactor;
  const int64_t output_groups = (channels + group_size - 1) / group_size;
  for (int stream = 0; stream < 4; ++stream) {
    const int64_t tokens = logical_tokens[stream];
    const int64_t packed_tokens = (tokens + kPackFactor - 1) / kPackFactor;
    const int64_t token_groups = (tokens + group_size - 1) / group_size;
    TORCH_CHECK(key_kernels[stream].is_cuda() && key_scaling_factors[stream].is_cuda() &&
                    key_zeros[stream].is_cuda() && value_kernels[stream].is_cuda() &&
                    value_scaling_factors[stream].is_cuda() && value_zeros[stream].is_cuda() &&
                    key_kernels[stream].scalar_type() == at::ScalarType::Int &&
                    value_kernels[stream].scalar_type() == at::ScalarType::Int &&
                    key_scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    key_zeros[stream].scalar_type() == at::ScalarType::Half &&
                    value_scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    value_zeros[stream].scalar_type() == at::ScalarType::Half,
                "fused sparse-attention stream dtypes differ from the CUDA ABI");
    TORCH_CHECK(key_kernels[stream].is_contiguous() &&
                    key_scaling_factors[stream].is_contiguous() &&
                    key_zeros[stream].is_contiguous() && value_kernels[stream].is_contiguous() &&
                    value_scaling_factors[stream].is_contiguous() &&
                    value_zeros[stream].is_contiguous(),
                "fused sparse-attention streams must be contiguous");
    TORCH_CHECK(key_kernels[stream].size(1) == packed_tokens &&
                    key_kernels[stream].size(2) == channels &&
                    key_scaling_factors[stream].size(1) == token_groups &&
                    key_scaling_factors[stream].size(2) == channels &&
                    key_scaling_factors[stream].sizes() == key_zeros[stream].sizes() &&
                    value_kernels[stream].size(1) == packed_output &&
                    value_kernels[stream].size(2) == tokens &&
                    value_scaling_factors[stream].size(1) == output_groups &&
                    value_scaling_factors[stream].size(2) == tokens &&
                    value_scaling_factors[stream].sizes() == value_zeros[stream].sizes(),
                "fused sparse-attention stream shape differs from its logical matrix");
  }
  auto output = torch::empty(
      {batch_heads, 1, channels}, query.options().dtype(torch::kFloat32));
  torch::Tensor current_logits;
  float* current_logits_ptr = nullptr;
  if (return_current_logits) {
    current_logits = torch::full(
        {batch_heads}, std::numeric_limits<float>::quiet_NaN(),
        query.options().dtype(torch::kFloat32));
    current_logits_ptr = current_logits.data_ptr<float>();
  }
  auto logits = torch::empty(
      {batch_heads, selected_tokens}, query.options().dtype(torch::kFloat32));
  auto probabilities = torch::empty_like(logits);
  constexpr int sparse_logit_threads = 128;
  constexpr int sparse_softmax_threads = 256;
  constexpr int sparse_pv_threads = 256;
  constexpr int sparse_logit_warps = sparse_logit_threads / kWarpSize;
  const dim3 sparse_logit_blocks(
      batch_heads, (selected_tokens + sparse_logit_warps - 1) / sparse_logit_warps);
  kivi_sparse_logits_delta4_tiled_kernel<<<
      sparse_logit_blocks, sparse_logit_threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __nv_bfloat16*>(query_bf16.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(current_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(key_kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(key_kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(key_kernels[2].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(key_kernels[3].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(key_scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_scaling_factors[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_scaling_factors[3].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[3].data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(), valid.data_ptr<bool>(), rank3.data_ptr<int32_t>(),
      rank4.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(tail_key.data_ptr<at::BFloat16>()),
      logits.data_ptr<float>(), current_logits_ptr,
      static_cast<int>(selected_tokens), static_cast<int>(logical_tokens[0]),
      static_cast<int>(logical_tokens[2]), static_cast<int>(logical_tokens[3]),
      static_cast<int>(tail_tokens), static_cast<int>(channels),
      static_cast<int>(group_size), static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(current_position),
      static_cast<float>(scaling), static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  kivi_attention_softmax_rows_kernel<<<
      batch_heads, sparse_softmax_threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), probabilities.data_ptr<float>(),
      static_cast<int>(selected_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 sparse_pv_blocks(batch_heads, packed_output);
  kivi_sparse_pv_delta4_tiled_kernel<<<
      sparse_pv_blocks, sparse_pv_threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      probabilities.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(current_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(value_kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(value_kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(value_kernels[2].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(value_kernels[3].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(value_scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_scaling_factors[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_scaling_factors[3].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[2].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[3].data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(), valid.data_ptr<bool>(), rank3.data_ptr<int32_t>(),
      rank4.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(tail_value.data_ptr<at::BFloat16>()),
      source_output.data_ptr<float>(), global_mass.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(selected_tokens), static_cast<int>(logical_tokens[0]),
      static_cast<int>(logical_tokens[2]), static_cast<int>(logical_tokens[3]),
      static_cast<int>(tail_tokens), static_cast<int>(channels),
      static_cast<int>(group_size), static_cast<int>(num_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(current_position),
      static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (return_current_logits) {
    return {output, current_logits};
  }
  return {output};
}

std::vector<torch::Tensor> dense_source_attention_cuda(
    torch::Tensor query,
    torch::Tensor query_bf16,
    torch::Tensor current_key,
    torch::Tensor current_value,
    std::vector<torch::Tensor> key_kernels,
    std::vector<torch::Tensor> key_scaling_factors,
    std::vector<torch::Tensor> key_zeros,
    std::vector<torch::Tensor> value_kernels,
    std::vector<torch::Tensor> value_scaling_factors,
    std::vector<torch::Tensor> value_zeros,
    torch::Tensor rank3,
    torch::Tensor rank4,
    torch::Tensor tail_key,
    torch::Tensor tail_value,
    double scaling,
    int64_t group_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t loop,
    std::vector<int64_t> logical_tokens,
    int64_t output_channels,
    bool return_logits) {
  TORCH_CHECK(key_kernels.size() == 4 && key_scaling_factors.size() == 4 &&
                  key_zeros.size() == 4 && value_kernels.size() == 4 &&
                  value_scaling_factors.size() == 4 && value_zeros.size() == 4 &&
                  logical_tokens.size() == 4,
              "fused dense source attention requires four K/V streams");
  TORCH_CHECK(query.is_cuda() && query_bf16.is_cuda() && current_key.is_cuda() &&
                  current_value.is_cuda() && rank3.is_cuda() && rank4.is_cuda() &&
                  tail_key.is_cuda() && tail_value.is_cuda(),
              "fused dense source-attention tensors must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half &&
                  query_bf16.scalar_type() == at::ScalarType::BFloat16 &&
                  current_key.scalar_type() == at::ScalarType::BFloat16 &&
                  current_value.scalar_type() == at::ScalarType::BFloat16 &&
                  tail_key.scalar_type() == at::ScalarType::BFloat16 &&
                  tail_value.scalar_type() == at::ScalarType::BFloat16 &&
                  rank3.scalar_type() == at::ScalarType::Int &&
                  rank4.scalar_type() == at::ScalarType::Int,
              "fused dense source-attention dtypes differ from the CUDA ABI");
  TORCH_CHECK(query.is_contiguous() && query_bf16.is_contiguous() &&
                  current_key.is_contiguous() && current_value.is_contiguous() &&
                  rank3.is_contiguous() && rank4.is_contiguous() &&
                  tail_key.is_contiguous() && tail_value.is_contiguous(),
              "fused dense source-attention tensors must be contiguous");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == 1 && query_bf16.dim() == 2 &&
                  current_key.dim() == 2 && current_value.sizes() == current_key.sizes() &&
                  tail_key.dim() == 3 && tail_value.sizes() == tail_key.sizes(),
              "invalid fused dense source-attention tensor ranks");
  TORCH_CHECK(group_size == 64 && loop >= 0 && loop < 2 && output_channels > 0 &&
                  logical_tokens[0] > 0 && logical_tokens[1] == logical_tokens[0],
              "invalid fused dense source-attention configuration");

  const int64_t batch_heads = query.size(0);
  const int64_t base_tokens = logical_tokens[0];
  const int64_t tail_tokens = tail_key.size(1);
  const int64_t selected_tokens = base_tokens + tail_tokens + 1;
  const int64_t current_position = base_tokens + tail_tokens;
  const int64_t channels = query.size(2);
  TORCH_CHECK(channels == output_channels &&
                  query_bf16.sizes() == current_key.sizes() &&
                  current_key.size(0) == batch_heads && current_key.size(1) == channels &&
                  tail_key.size(0) == batch_heads && tail_key.size(2) == channels &&
                  rank3.numel() == base_tokens && rank4.numel() == base_tokens,
              "fused dense source-attention shapes differ from the anchor");

  const int64_t packed_output = (channels + kPackFactor - 1) / kPackFactor;
  const int64_t output_groups = (channels + group_size - 1) / group_size;
  for (int stream = 0; stream < 4; ++stream) {
    const int64_t tokens = logical_tokens[stream];
    const int64_t packed_tokens = (tokens + kPackFactor - 1) / kPackFactor;
    const int64_t token_groups = (tokens + group_size - 1) / group_size;
    TORCH_CHECK(key_kernels[stream].is_cuda() && key_scaling_factors[stream].is_cuda() &&
                    key_zeros[stream].is_cuda() && value_kernels[stream].is_cuda() &&
                    value_scaling_factors[stream].is_cuda() && value_zeros[stream].is_cuda() &&
                    key_kernels[stream].scalar_type() == at::ScalarType::Int &&
                    value_kernels[stream].scalar_type() == at::ScalarType::Int &&
                    key_scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    key_zeros[stream].scalar_type() == at::ScalarType::Half &&
                    value_scaling_factors[stream].scalar_type() == at::ScalarType::Half &&
                    value_zeros[stream].scalar_type() == at::ScalarType::Half,
                "fused dense source-attention stream dtypes differ from the CUDA ABI");
    TORCH_CHECK(key_kernels[stream].is_contiguous() &&
                    key_scaling_factors[stream].is_contiguous() &&
                    key_zeros[stream].is_contiguous() && value_kernels[stream].is_contiguous() &&
                    value_scaling_factors[stream].is_contiguous() &&
                    value_zeros[stream].is_contiguous(),
                "fused dense source-attention streams must be contiguous");
    TORCH_CHECK(key_kernels[stream].size(1) == packed_tokens &&
                    key_kernels[stream].size(2) == channels &&
                    key_scaling_factors[stream].size(1) == token_groups &&
                    key_scaling_factors[stream].size(2) == channels &&
                    key_scaling_factors[stream].sizes() == key_zeros[stream].sizes() &&
                    value_kernels[stream].size(1) == packed_output &&
                    value_kernels[stream].size(2) == tokens &&
                    value_scaling_factors[stream].size(1) == output_groups &&
                    value_scaling_factors[stream].size(2) == tokens &&
                    value_scaling_factors[stream].sizes() == value_zeros[stream].sizes(),
                "fused dense source-attention stream shape differs from its logical matrix");
  }

  auto output = torch::empty(
      {batch_heads, 1, channels}, query.options().dtype(torch::kFloat32));
  auto logits = torch::empty(
      {batch_heads, 1, selected_tokens}, query.options().dtype(torch::kFloat32));
  auto probabilities = torch::empty_like(logits);
  constexpr int logit_threads = 128;
  constexpr int softmax_threads = 256;
  constexpr int pv_threads = 256;
  constexpr int logit_warps = logit_threads / kWarpSize;
  const dim3 logit_blocks(
      batch_heads, (selected_tokens + logit_warps - 1) / logit_warps);
  kivi_attention_logits_delta4_tiled_kernel<<<
      logit_blocks, logit_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __nv_bfloat16*>(query_bf16.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(current_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(key_kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(key_kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(key_scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const __nv_bfloat16*>(tail_key.data_ptr<at::BFloat16>()),
      logits.data_ptr<float>(), static_cast<int>(selected_tokens),
      static_cast<int>(base_tokens), static_cast<int>(tail_tokens),
      static_cast<int>(channels), static_cast<int>(group_size),
      static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(current_position), static_cast<float>(scaling),
      static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  kivi_attention_softmax_rows_kernel<<<
      batch_heads, softmax_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), probabilities.data_ptr<float>(),
      static_cast<int>(selected_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 pv_blocks(batch_heads, packed_output);
  kivi_attention_pv_delta4_tiled_kernel<<<
      pv_blocks, pv_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      probabilities.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(current_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(value_kernels[0].data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(value_kernels[1].data_ptr<int32_t>()),
      reinterpret_cast<const half*>(value_scaling_factors[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_scaling_factors[1].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[0].data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_zeros[1].data_ptr<at::Half>()),
      reinterpret_cast<const __nv_bfloat16*>(tail_value.data_ptr<at::BFloat16>()),
      output.data_ptr<float>(),
      static_cast<int>(selected_tokens), static_cast<int>(base_tokens),
      static_cast<int>(tail_tokens), static_cast<int>(channels),
      static_cast<int>(group_size),
      static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(current_position), static_cast<int>(loop));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (return_logits) {
    return {output, logits};
  }
  return {output};
}
