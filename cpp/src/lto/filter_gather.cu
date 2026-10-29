/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/cache.hpp"
#include "jit/row_ir.hpp"
#include "jit/udf_reflection.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lto/codec.cuh>
#include <cudf/lto/filter_gather.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <algorithm>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cudf {
namespace lto {
namespace {

constexpr int block_threads = 256;

/// C++ spellings of a type in generated code.
struct type_names {
  std::string logical;    ///< The type
  std::string storage;    ///< Its device storage representation
  std::string construct;  ///< Expression building the type from a storage value named `r`
};

template <typename Rep>
std::string rep_name()
{
  if constexpr (std::is_same_v<Rep, __int128_t>) {
    return "__int128_t";
  } else {
    return type_to_name(data_type{type_to_id<Rep>()});
  }
}

struct type_names_fn {
  template <typename T>
  type_names operator()(data_type type) const
    requires(is_fixed_width<T>())
  {
    auto logical = type_to_name(type);
    if constexpr (is_fixed_point<T>()) {
      auto storage = rep_name<typename T::rep>();
      return {logical,
              storage,
              std::format("{}{{numeric::scaled_integer<{}>{{r, numeric::scale_type{{{}}}}}}}",
                          logical,
                          storage,
                          type.scale())};
    } else if constexpr (is_timestamp<T>()) {
      return {logical,
              rep_name<typename T::rep>(),
              std::format("{}{{{}::duration{{r}}}}", logical, logical)};
    } else if constexpr (is_duration<T>()) {
      return {logical, rep_name<typename T::rep>(), std::format("{}{{r}}", logical)};
    } else {
      return {logical, logical, "r"};
    }
  }

  template <typename T>
  type_names operator()(data_type) const
    requires(!is_fixed_width<T>())
  {
    CUDF_FAIL("filter_gather supports only fixed-width columns", std::invalid_argument);
  }
};

type_names names_of(data_type type) { return type_dispatcher(type, type_names_fn{}, type); }

data_type type_of(filter_input const& input)
{
  if (auto const* col = std::get_if<column_view>(&input)) { return col->type(); }
  return std::get<encoded_column_view>(input).type;
}

size_type size_of_input(filter_input const& input)
{
  if (auto const* col = std::get_if<column_view>(&input)) { return col->size(); }
  return std::get<encoded_column_view>(input).size;
}

/// Device pointer to the first element of a fixed-width column.
void const* data_of(column_view const& col)
{
  return col.head<uint8_t>() + static_cast<size_t>(col.offset()) * size_of(col.type());
}

/// Assigns each distinct codec load hook a declaration.
struct codec_symbols {
  std::map<std::string, std::string> declarations;

  void add(column_codec const& codec)
  {
    declarations.try_emplace(
      codec.load,
      std::format(
        "extern \"C\" __device__ void {}(void*, cudf::lto::codec_scope const*, long long, "
        "long long);\n",
        codec.load));
  }
};

/// Emits a statement declaring `var` (of the storage type) with the value of input `i` at `row`.
std::string emit_load(std::vector<filter_input> const& inputs,
                      size_t i,
                      std::string const& var,
                      std::string const& row)
{
  auto names = names_of(type_of(inputs[i]));
  if (auto const* enc = std::get_if<encoded_column_view>(&inputs[i])) {
    return std::format(
      "{0} {1}; {2}(&{1}, &scopes[{3}], {4}, 1);\n", names.storage, var, enc->codec->load, i, row);
  }
  return std::format(
    "{0} {1} = static_cast<{0} const*>(plain[{2}])[{3}];\n", names.storage, var, i, row);
}

constexpr char const* kernel_prelude = R"***(
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lto/codec.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

)***";

constexpr char const* kernel_entry = R"***(
extern "C" __global__ void cudf_kernel_entry(int phase,
                                             long long num_rows,
                                             cudf::lto::codec_scope const* scopes,
                                             void const* const* plain,
                                             void const* const* scalars,
                                             unsigned* mask,
                                             long long* block_offsets,
                                             void* const* outputs)
{
  constexpr int block_threads = 256;
  constexpr int block_warps   = block_threads / 32;
  __shared__ int warp_offsets[block_warps];

  long long const row = static_cast<long long>(blockIdx.x) * block_threads + threadIdx.x;
  int const lane      = threadIdx.x % 32;
  int const warp      = threadIdx.x / 32;
  long long const word_index = static_cast<long long>(blockIdx.x) * block_warps + warp;

  if (phase == 2) {
    if (row < num_rows) { cudf_lto_gather(row, row, scopes, plain, outputs); }
    return;
  }

  if (phase == 0) {
    bool const keep = row < num_rows && cudf_lto_eval(row, scopes, plain, scalars);
    unsigned const ballot = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) { mask[word_index] = ballot; }
    int const count = __syncthreads_count(keep);
    if (threadIdx.x == 0) { block_offsets[blockIdx.x] = count; }
    return;
  }

  unsigned const word = mask[word_index];
  if (lane == 0) { warp_offsets[warp] = __popc(word); }
  __syncthreads();
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int w = 0; w < block_warps; ++w) {
      int const count = warp_offsets[w];
      warp_offsets[w] = sum;
      sum += count;
    }
  }
  __syncthreads();
  if ((word >> lane) & 1u) {
    long long const position =
      block_offsets[blockIdx.x] + warp_offsets[warp] + __popc(word & ((1u << lane) - 1u));
    cudf_lto_gather(row, position, scopes, plain, outputs);
  }
}
)***";

/// Generates the CUDA source of the fused kernel.
std::string generate_kernel(std::vector<filter_input> const& inputs,
                            cudf::detail::row_ir::transform_args const* predicate,
                            std::span<size_type const> gather_map)
{
  codec_symbols symbols;
  for (auto const& input : inputs) {
    if (auto const* enc = std::get_if<encoded_column_view>(&input)) { symbols.add(*enc->codec); }
  }

  std::string source = kernel_prelude;
  for (auto const& [_, declaration] : symbols.declarations) {
    source += declaration;
  }
  source += "\n";
  if (predicate != nullptr) { source += predicate->udf; }
  source += R"***(

namespace {

__device__ bool cudf_lto_eval(long long row,
                              cudf::lto::codec_scope const* scopes,
                              void const* const* plain,
                              void const* const* scalars)
{
)***";

  auto const num_predicate_inputs = predicate != nullptr ? predicate->inputs.size() : 0;
  auto const null_aware_predicate =
    predicate != nullptr && predicate->is_null_aware == null_aware::YES;
  auto const optional = [&](std::string const& type) {
    return null_aware_predicate ? std::format("cuda::std::optional<{}>", type) : type;
  };
  std::string call_args = "&keep";
  for (size_t k = 0; k < num_predicate_inputs; ++k) {
    auto names =
      std::visit([](auto const& in) { return names_of(in.type()); }, predicate->inputs[k]);
    auto const load =
      predicate->input_column_indices[k].has_value()
        ? emit_load(inputs, static_cast<size_t>(*predicate->input_column_indices[k]), "r", "row")
        : std::format("{0} r = *static_cast<{0} const*>(scalars[{1}]);\n", names.storage, k);
    source += std::format("  {} in{} = [&] {{\n    {}    return {};\n  }}();\n",
                          optional(names.logical),
                          k,
                          load,
                          names.construct);
    call_args += std::format(", in{}", k);
  }
  if (predicate != nullptr) {
    source += std::format("  {} keep{{}};\n", optional("bool"));
    source += std::format("  cudf_lto_predicate({});\n", call_args);
  } else {
    source += "  bool keep = true;\n";
  }
  source += null_aware_predicate ? "  return keep.value_or(false);\n}" : "  return keep;\n}";
  source += R"***(

__device__ void cudf_lto_gather(long long row,
                                long long position,
                                cudf::lto::codec_scope const* scopes,
                                void const* const* plain,
                                void* const* outputs)
{
)***";
  for (size_t g = 0; g < gather_map.size(); ++g) {
    auto const i = static_cast<size_t>(gather_map[g]);
    auto names   = names_of(type_of(inputs[i]));
    source += "  {\n  ";
    source += emit_load(inputs, i, "r", "row");
    source +=
      std::format("  static_cast<{} *>(outputs[{}])[position] = r;\n  }}\n", names.storage, g);
  }
  source += "}\n\n}  // namespace\n";
  source += kernel_entry;
  return source;
}

}  // namespace

std::shared_ptr<lto_fragment const> compile_lto_ir(std::string const& name,
                                                   std::string const& source)
{
  CUDF_FUNC_RANGE();
  auto blob = get_source_fragment(name, source, {}, {});
  auto view = blob->view();
  return std::make_shared<lto_fragment const>(
    lto_fragment{std::vector<std::uint8_t>(view.begin(), view.end()), lto_binary_type::LTO_IR});
}

namespace {

void validate(std::vector<filter_input> const& inputs, std::span<size_type const> gather_map)
{
  CUDF_EXPECTS(!inputs.empty(), "LTO decoding requires at least one input", std::invalid_argument);
  auto const num_rows = size_of_input(inputs.front());
  for (auto const& input : inputs) {
    CUDF_EXPECTS(size_of_input(input) == num_rows,
                 "LTO decoding inputs must have the same size",
                 std::invalid_argument);
    CUDF_EXPECTS(is_fixed_width(type_of(input)),
                 "LTO decoding supports only fixed-width columns",
                 std::invalid_argument);
    if (auto const* col = std::get_if<column_view>(&input)) {
      CUDF_EXPECTS(
        !col->has_nulls(), "LTO decoding inputs must not have nulls", std::invalid_argument);
    } else {
      auto const& enc = std::get<encoded_column_view>(input);
      CUDF_EXPECTS(enc.codec != nullptr && !enc.codec->load.empty(),
                   "Encoded inputs require a codec with a load hook",
                   std::invalid_argument);
    }
  }
  CUDF_EXPECTS(!gather_map.empty(),
               "LTO decoding requires at least one gathered column",
               std::invalid_argument);
  for (auto index : gather_map) {
    CUDF_EXPECTS(index >= 0 && static_cast<size_t>(index) < inputs.size(),
                 "LTO decoding gather index out of range",
                 std::out_of_range);
  }
}

kernel link_kernel(std::vector<filter_input> const& inputs, std::string const& source)
{
  auto kernel_fragment = get_source_fragment("cudf/lto/filter_gather.cu", source, {}, {});

  std::vector<rtcx::memory_fragment> fragments{
    {.data = kernel_fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};
  std::vector<lto_fragment const*> linked;
  for (auto const& input : inputs) {
    auto const* enc = std::get_if<encoded_column_view>(&input);
    if (enc == nullptr) { continue; }
    for (auto const& fragment : enc->codec->fragments) {
      if (std::ranges::find(linked, fragment.get()) != linked.end()) { continue; }
      linked.push_back(fragment.get());
      fragments.push_back({.data = fragment->data,
                           .type = jit::as_rtcx_binary_type(fragment->type),
                           .name = nullptr});
    }
  }
  return get_lto_linked_kernel("cudf/lto/filter_gather.cu", {}, fragments);
}

/// Filters (when `predicate` is set) and gathers `inputs`; see `filter_gather`.
std::unique_ptr<table> run(std::span<filter_input const> input_span,
                           ast::expression const* predicate,
                           std::span<size_type const> gather_map,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  std::vector<filter_input> inputs(input_span.begin(), input_span.end());
  validate(inputs, gather_map);
  auto const num_rows = size_of_input(inputs.front());
  auto const temp_mr  = cudf::get_current_device_resource_ref();

  auto make_outputs = [&](size_type size) {
    std::vector<std::unique_ptr<column>> outputs;
    for (auto index : gather_map) {
      outputs.emplace_back(
        make_fixed_width_column(type_of(inputs[index]), size, mask_state::UNALLOCATED, stream, mr));
    }
    return outputs;
  };

  // The predicate is generated against empty placeholder columns that carry only the input types.
  std::optional<cudf::detail::row_ir::transform_args> args;
  if (predicate != nullptr) {
    std::vector<column_view> placeholders;
    for (auto const& input : inputs) {
      placeholders.emplace_back(type_of(input), 0, nullptr, nullptr, 0);
    }
    args = cudf::detail::row_ir::ast_converter::filter(cudf::detail::row_ir::target::CUDA,
                                                       *predicate,
                                                       table_view{placeholders},
                                                       table_view{},
                                                       "cudf_lto_predicate",
                                                       stream,
                                                       temp_mr);
  }

  if (num_rows == 0) { return std::make_unique<table>(make_outputs(0)); }

  auto kernel = link_kernel(inputs, generate_kernel(inputs, args ? &*args : nullptr, gather_map));

  std::vector<codec_scope> h_scopes(inputs.size(), codec_scope{nullptr, nullptr});
  std::vector<void const*> h_plain(inputs.size(), nullptr);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (auto const* col = std::get_if<column_view>(&inputs[i])) {
      h_plain[i] = data_of(*col);
    } else {
      auto const& enc = std::get<encoded_column_view>(inputs[i]);
      h_scopes[i]     = codec_scope{enc.column_data, enc.user_data};
    }
  }
  std::vector<void const*> h_scalars(args ? args->inputs.size() : 0, nullptr);
  for (size_t k = 0; k < h_scalars.size(); ++k) {
    if (auto const* s = std::get_if<scalar_column_view>(&args->inputs[k])) {
      h_scalars[k] = data_of(s->as_column_view());
    }
  }

  auto const num_blocks = cudf::util::div_rounding_up_safe<int64_t>(num_rows, block_threads);
  auto d_scopes         = cudf::detail::make_device_uvector_async(h_scopes, stream, temp_mr);
  auto d_plain          = cudf::detail::make_device_uvector_async(h_plain, stream, temp_mr);
  auto d_scalars        = cudf::detail::make_device_uvector_async(h_scalars, stream, temp_mr);

  long long rows          = num_rows;
  auto const* scopes_ptr  = d_scopes.data();
  auto const* plain_ptr   = d_plain.data();
  auto const* scalars_ptr = d_scalars.data();
  auto const grid         = rtcx::cuda_dim3{static_cast<uint32_t>(num_blocks)};
  auto const block        = rtcx::cuda_dim3{static_cast<uint32_t>(block_threads)};
  auto launch = [&](int phase, unsigned* mask, long long* offsets, void* const* outputs) {
    kernel.launch_with(grid,
                       block,
                       0,
                       stream,
                       phase,
                       rows,
                       scopes_ptr,
                       plain_ptr,
                       scalars_ptr,
                       mask,
                       offsets,
                       outputs);
  };

  auto device_pointers = [&](std::vector<std::unique_ptr<column>>& outputs) {
    std::vector<void*> h_outputs;
    for (auto& output : outputs) {
      h_outputs.push_back(output->mutable_view().head());
    }
    return cudf::detail::make_device_uvector_async(h_outputs, stream, temp_mr);
  };

  if (predicate == nullptr) {
    auto outputs   = make_outputs(num_rows);
    auto d_outputs = device_pointers(outputs);
    launch(2, nullptr, nullptr, d_outputs.data());
    return std::make_unique<table>(std::move(outputs));
  }

  rmm::device_uvector<unsigned> mask(num_blocks * (block_threads / 32), stream, temp_mr);
  rmm::device_uvector<long long> block_offsets(num_blocks + 1, stream, temp_mr);
  launch(0, mask.data(), block_offsets.data(), nullptr);

  CUDF_CUDA_TRY(
    cudaMemsetAsync(block_offsets.data() + num_blocks, 0, sizeof(long long), stream.get()));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         block_offsets.begin(),
                         block_offsets.end(),
                         block_offsets.begin());
  auto const total = block_offsets.element(num_blocks, stream);

  auto outputs = make_outputs(static_cast<size_type>(total));
  if (total > 0) {
    auto d_outputs = device_pointers(outputs);
    launch(1, mask.data(), block_offsets.data(), d_outputs.data());
  }
  return std::make_unique<table>(std::move(outputs));
}

}  // namespace

std::unique_ptr<table> filter_gather(std::span<filter_input const> inputs,
                                     ast::expression const& predicate,
                                     std::span<size_type const> gather_map,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return run(inputs, &predicate, gather_map, stream, mr);
}

std::unique_ptr<table> decode(std::span<filter_input const> inputs,
                              std::span<size_type const> gather_map,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return run(inputs, nullptr, gather_map, stream, mr);
}

}  // namespace lto
}  // namespace cudf
