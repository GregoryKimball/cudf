/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/lto/codecs/for_bitpack.hpp>
#include <cudf/lto/filter_gather.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

__device__ uint64_t mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/// A column of `n` values drawn from [lo, hi], with a run-length of `cluster` rows per value step
/// so that `cluster` > 1 mimics a sorted (clustered) column such as l_shipdate.
template <typename Rep>
std::unique_ptr<cudf::column> make_column(
  cudf::data_type type, cudf::size_type n, int64_t lo, int64_t hi, int64_t cluster, uint64_t seed)
{
  auto col  = cudf::make_fixed_width_column(type, n);
  auto* out = col->mutable_view().data<Rep>();
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    thrust::counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(n),
                    out,
                    [=] __device__(int64_t i) {
                      auto const span = static_cast<uint64_t>(hi - lo + 1);
                      if (cluster > 1) { return static_cast<Rep>(lo + (i / cluster) % span); }
                      return static_cast<Rep>(lo + mix(i ^ seed) % span);
                    });
  return col;
}

enum class residency { DEVICE, PINNED, PAGEABLE };

residency residency_from_string(std::string const& name)
{
  if (name == "device") { return residency::DEVICE; }
  if (name == "pinned") { return residency::PINNED; }
  if (name == "pageable") { return residency::PAGEABLE; }
  CUDF_FAIL("unknown residency " + name);
}

/// Host memory that the GPU reads directly over the CPU-GPU link: pinned, or pageable system
/// memory such as GQE's shared-segment storage.
struct host_buffer {
  void* ptr = nullptr;
  bool pinned;
  host_buffer(std::size_t size, bool pin) : pinned{pin}
  {
    if (pinned) {
      CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
    } else {
      ptr = std::aligned_alloc(4096, cudf::util::round_up_safe<std::size_t>(size, 4096));
      CUDF_EXPECTS(ptr != nullptr, "pageable allocation failed");
    }
  }
  host_buffer(host_buffer&& other) noexcept
    : ptr{std::exchange(other.ptr, nullptr)}, pinned{other.pinned}
  {
  }
  host_buffer(host_buffer const&)            = delete;
  host_buffer& operator=(host_buffer const&) = delete;
  host_buffer& operator=(host_buffer&&)      = delete;
  ~host_buffer()
  {
    if (ptr == nullptr) { return; }
    if (pinned) {
      cudaFreeHost(ptr);
    } else {
      std::free(ptr);
    }
  }
};

/// A buffer of `size` bytes in device, pinned host, or pageable host memory.
struct resident_buffer {
  std::size_t size;
  std::optional<rmm::device_buffer> device;
  std::optional<host_buffer> host;

  resident_buffer(void const* src, std::size_t bytes, residency where, cuda::stream_ref stream)
    : size{bytes}
  {
    void* dst = nullptr;
    if (where != residency::DEVICE) {
      dst = host.emplace(bytes, where == residency::PINNED).ptr;
    } else {
      dst = device.emplace(bytes, stream).data();
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream.get()));
    stream.sync();
  }
  [[nodiscard]] void const* data() const { return host ? host->ptr : device->data(); }
};

struct q6_data {
  std::vector<cudf::data_type> types;
  std::vector<resident_buffer> plain;
  std::vector<resident_buffer> encoded;
  std::vector<cudf::column_view> plain_views;
  std::vector<cudf::lto::filter_input> plain_inputs;
  std::vector<cudf::lto::filter_input> encoded_inputs;
  cudf::size_type num_rows;

  [[nodiscard]] std::size_t plain_bytes() const
  {
    std::size_t bytes = 0;
    for (auto const& b : plain) {
      bytes += b.size;
    }
    return bytes;
  }
  [[nodiscard]] std::size_t encoded_bytes() const
  {
    std::size_t bytes = 0;
    for (auto const& b : encoded) {
      bytes += b.size;
    }
    return bytes;
  }
};

/// TPC-H Q6 columns: l_shipdate (clustered dates), l_discount, l_quantity, l_extendedprice (all
/// DECIMAL64 with scale -2), sized and distributed like lineitem, with the plain and encoded data
/// placed `where`.
q6_data make_q6(cudf::size_type n, residency where)
{
  auto const stream = cudf::get_default_stream();
  q6_data d;
  d.num_rows     = n;
  auto const dec = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  std::vector<std::unique_ptr<cudf::column>> columns;
  // Seven years of shipdates spread over the rows.
  columns.push_back(make_column<int32_t>(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                         n,
                                         8036,
                                         10591,
                                         std::max<int64_t>(1, n / 2556),
                                         1));
  columns.push_back(make_column<int64_t>(dec, n, 0, 10, 1, 2));
  columns.push_back(make_column<int64_t>(dec, n, 100, 5000, 1, 3));
  columns.push_back(make_column<int64_t>(dec, n, 90000, 10500000, 1, 4));
  for (auto& col : columns) {
    auto const bytes = static_cast<std::size_t>(n) * cudf::size_of(col->type());
    auto encoded     = cudf::lto::codecs::for_bitpack_encode(col->view());
    d.types.push_back(col->type());
    d.plain.emplace_back(col->view().head(), bytes, where, stream);
    d.encoded.emplace_back(encoded.data(), encoded.size(), where, stream);
  }
  for (std::size_t i = 0; i < columns.size(); ++i) {
    d.plain_views.emplace_back(d.types[i], n, d.plain[i].data(), nullptr, 0);
    d.plain_inputs.emplace_back(d.plain_views.back());
    d.encoded_inputs.emplace_back(
      cudf::lto::encoded_column_view{.type        = d.types[i],
                                     .size        = n,
                                     .column_data = d.encoded[i].data(),
                                     .codec       = &cudf::lto::codecs::for_bitpack_codec()});
  }
  return d;
}

/// Copies `buffers` to device memory in bulk, as a read task stages host-resident data.
std::vector<rmm::device_buffer> stage(std::vector<resident_buffer> const& buffers,
                                      cuda::stream_ref stream)
{
  std::vector<rmm::device_buffer> staged;
  for (auto const& b : buffers) {
    staged.emplace_back(b.data(), b.size, stream);
  }
  return staged;
}

/// The Q6 predicate. With `column_literals`, literals are one-row columns referenced through
/// `scalar_column_view`, which spares the JIT converter from materializing a column per scalar on
/// every call.
struct q6_predicate {
  cudf::timestamp_scalar<cudf::timestamp_D> date_lo{cudf::duration_D{8766}, true};
  cudf::timestamp_scalar<cudf::timestamp_D> date_hi{cudf::duration_D{9131}, true};
  cudf::fixed_point_scalar<numeric::decimal64> disc_lo{5, numeric::scale_type{-2}};
  cudf::fixed_point_scalar<numeric::decimal64> disc_hi{7, numeric::scale_type{-2}};
  cudf::fixed_point_scalar<numeric::decimal64> qty_hi{2400, numeric::scale_type{-2}};
  std::vector<std::unique_ptr<cudf::column>> literal_columns;
  cudf::ast::tree tree;
  cudf::ast::expression const* root;

  explicit q6_predicate(bool column_literals)
  {
    using op     = cudf::ast::ast_operator;
    auto literal = [&](auto& value) -> cudf::ast::expression const& {
      if (!column_literals) { return tree.push(cudf::ast::literal(value)); }
      literal_columns.push_back(cudf::make_column_from_scalar(value, 1));
      return tree.push(
        cudf::ast::literal(cudf::scalar_column_view{literal_columns.back()->view()}));
    };
    auto& date = tree.push(cudf::ast::column_reference(0));
    auto& disc = tree.push(cudf::ast::column_reference(1));
    auto& qty  = tree.push(cudf::ast::column_reference(2));
    auto& c1   = tree.push(cudf::ast::operation(op::GREATER_EQUAL, date, literal(date_lo)));
    auto& c2   = tree.push(cudf::ast::operation(op::LESS, date, literal(date_hi)));
    auto& c3   = tree.push(cudf::ast::operation(op::GREATER_EQUAL, disc, literal(disc_lo)));
    auto& c4   = tree.push(cudf::ast::operation(op::LESS_EQUAL, disc, literal(disc_hi)));
    auto& c5   = tree.push(cudf::ast::operation(op::LESS, qty, literal(qty_hi)));
    auto& a1   = tree.push(cudf::ast::operation(op::LOGICAL_AND, c1, c2));
    auto& a2   = tree.push(cudf::ast::operation(op::LOGICAL_AND, c3, c4));
    auto& a3   = tree.push(cudf::ast::operation(op::LOGICAL_AND, a1, a2));
    root       = &tree.push(cudf::ast::operation(op::LOGICAL_AND, a3, c5));
  }
};

}  // namespace

void bench_q6_filter_gather(nvbench::state& state)
{
  auto const n      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const method          = state.get_string("method");
  auto const column_literals = state.get_string("literals") == "column";
  if (column_literals && method.find("cudf") != std::string::npos) {
    state.skip("cudf::compute_column supports only scalar literals");
    return;
  }
  auto data = make_q6(n, residency_from_string(state.get_string("residency")));
  q6_predicate predicate{column_literals};
  std::vector<cudf::size_type> const gather_map{3, 1};

  cudf::table_view plain_table{data.plain_views};
  auto const stream = cudf::get_default_stream();
  auto cudf_filter  = [&](cudf::table_view table) {
    auto mask = cudf::compute_column(table, *predicate.root, stream);
    return cudf::apply_retention_mask(table.select(gather_map), mask->view(), stream);
  };
  auto run = [&]() -> std::unique_ptr<cudf::table> {
    if (method == "cudf_plain") { return cudf_filter(plain_table); }
    if (method == "staged_cudf_plain") {
      auto staged = stage(data.plain, stream);
      std::vector<cudf::column_view> views;
      for (std::size_t i = 0; i < staged.size(); ++i) {
        views.emplace_back(data.types[i], n, staged[i].data(), nullptr, 0);
      }
      return cudf_filter(cudf::table_view{views});
    }
    if (method == "staged_fused_encoded") {
      auto staged = stage(data.encoded, stream);
      std::vector<cudf::lto::filter_input> inputs;
      for (std::size_t i = 0; i < staged.size(); ++i) {
        inputs.emplace_back(
          cudf::lto::encoded_column_view{.type        = data.types[i],
                                         .size        = n,
                                         .column_data = staged[i].data(),
                                         .codec       = &cudf::lto::codecs::for_bitpack_codec()});
      }
      return cudf::lto::filter_gather(inputs, *predicate.root, gather_map, stream);
    }
    if (method == "fused_plain") {
      return cudf::lto::filter_gather(data.plain_inputs, *predicate.root, gather_map, stream);
    }
    if (method == "fused_encoded") {
      return cudf::lto::filter_gather(data.encoded_inputs, *predicate.root, gather_map, stream);
    }
    if (method == "decode_then_cudf") {
      std::vector<cudf::size_type> const all{0, 1, 2, 3};
      auto decoded = cudf::lto::decode(data.encoded_inputs, all, stream);
      return cudf_filter(decoded->view());
    }
    CUDF_FAIL("unknown method " + method);
  };

  auto const rows_out = run()->num_rows();
  state.add_element_count(n);
  state.add_global_memory_reads<int8_t>(method.find("encoded") != std::string::npos ||
                                            method == "decode_then_cudf"
                                          ? data.encoded_bytes()
                                          : data.plain_bytes());
  state.add_summary("rows_out").set_int64("value", rows_out);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { run(); });
}

NVBENCH_BENCH(bench_q6_filter_gather)
  .set_name("lto_q6_filter_gather")
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 64'000'000})
  .add_string_axis("method",
                   {"cudf_plain",
                    "staged_cudf_plain",
                    "fused_plain",
                    "fused_encoded",
                    "staged_fused_encoded",
                    "decode_then_cudf"})
  .add_string_axis("literals", {"scalar", "column"})
  .add_string_axis("residency", {"device", "pinned", "pageable"});
