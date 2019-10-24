/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/substring.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"


namespace cudf
{
namespace strings
{
namespace detail
{

namespace
{

/**
 * @brief Used as template parameter to divide size calculation from
 * the actual string operation within a function.
 * Useful when most of the logic is identical for both passes.
 */
enum TwoPass
{
    SizeOnly = 0, ///< calculate the size only
    ExecuteOp     ///< run the string operation
};

/**
 * @brief Function logic for the substring API.
 * This will perform a substring operation on each string
 * using the provided start, stop, and step parameters.
 */
template <TwoPass Pass=SizeOnly>
struct substring_fn
{
    const column_device_view d_column;
    int32_t start, stop, step;
    const int32_t* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_column.is_null(idx) )
            return 0; // null string
        string_view d_str = d_column.template element<string_view>(idx);
        auto length = d_str.length();
        if( start >= length )
            return 0; // empty string
        auto itr = d_str.begin() + start;
        auto end_itr = d_str.begin() + (((stop<0) || (stop>length) ) ? length : stop);
        size_type bytes = 0;
        char* d_buffer = nullptr;
        if( Pass==ExecuteOp )
            d_buffer = d_chars + d_offsets[idx];
        while( itr < end_itr )
        {
            if( Pass==SizeOnly )
                bytes += detail::bytes_in_char_utf8(*itr);
            else
                d_buffer += from_char_utf8(*itr,d_buffer);
            itr += step;
        }
        return bytes;
    }
};

} // namespace

// Returns strings column of substrings each string in the provided column.
std::unique_ptr<cudf::column> substring( strings_column_view strings,
                                         int32_t start, int32_t stop=-1, int32_t step=1,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( start >= 0, "Parameter start must be zero or positive integer.");
    if( step == 0 )
        step = 1;
    CUDF_EXPECTS( step > 0, "Parameter step must be positive integer.");
    if( (stop > 0) && (start > stop) )
        CUDF_FAIL("Invalid start or stop parameter value.");

    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    cudf::size_type null_count = d_column.null_count();
    if( d_column.nullable() )
        null_mask = rmm::device_buffer( d_column.null_mask(),
                                        gdf_valid_allocation_size(strings_count),
                                        stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        substring_fn<SizeOnly>{d_column, start, stop, step} );
    auto offsets_column = detail::make_offsets_child_column(offsets_transformer_itr,
                                               offsets_transformer_itr+strings_count,
                                               mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_new_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
        substring_fn<ExecuteOp>{d_column, start, stop, step, d_new_offsets, d_chars} );
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

namespace
{

/**
 * @brief Function logic for substring_from API.
 * This does both calculate and the execute based on template parameter.
 *
 */
template <typename PositionType, TwoPass Pass=SizeOnly>
struct substring_from_fn
{
    const column_device_view d_column;
    const PositionType* starts;
    const PositionType* stops;
    const int32_t* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_column.is_null(idx) )
            return 0; // null string
        string_view d_str = d_column.template element<string_view>(idx);
        size_type length = d_str.length();
        size_type start = static_cast<size_type>(starts[idx]);
        if( start >= length )
            return 0; // empty string
        size_type stop = static_cast<size_type>(stops[idx]);
        size_type end = (((stop<0) || (stop>length) ) ? length : stop);
        string_view d_substr = d_str.substr(start,end-start);
        if( Pass==SizeOnly )
            return d_substr.size_bytes();
        else
        {
            memcpy( d_chars + d_offsets[idx], d_substr.data(), d_substr.size_bytes() );
            return 0;
        }
    }
};

/**
 * @brief Called by the type-dispatcher for resolving the position columns
 * (starts_column and stops_column) to actual types.
 */
struct dispatch_substring_from_fn
{
    template <typename PositionType, std::enable_if_t<std::is_integral<PositionType>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()( strings_column_view& strings,
                                              column_view& starts_column, column_view& stops_column,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream ) const noexcept
    {
        const PositionType* starts = starts_column.data<PositionType>();
        const PositionType* stops = stops_column.data<PositionType>();

        size_type strings_count = strings.size();
        auto execpol = rmm::exec_policy(0);
        auto strings_column = column_device_view::create(strings.parent(),stream);
        auto d_column = *strings_column;

        // copy the null mask
        rmm::device_buffer null_mask;
        cudf::size_type null_count = d_column.null_count();
        if( d_column.nullable() )
            null_mask = rmm::device_buffer( d_column.null_mask(),
                                            gdf_valid_allocation_size(strings_count),
                                            stream, mr);
        // build offsets column
        auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<PositionType>(0),
            substring_from_fn<PositionType>{d_column,starts,stops} );
        auto offsets_column = detail::make_offsets_child_column(offsets_transformer_itr,
                                                                offsets_transformer_itr+strings_count,
                                                                mr, stream);
        auto offsets_view = offsets_column->view();
        auto d_new_offsets = offsets_view.template data<int32_t>();

        // build chars column
        size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
        auto chars_column = strings::detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
        auto chars_view = chars_column->mutable_view();
        auto d_chars = chars_view.template data<char>();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
            substring_from_fn<PositionType,ExecuteOp>{d_column, starts, stops, d_new_offsets, d_chars});
        //
        return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                                   null_count, std::move(null_mask), stream, mr);
    }
    //
    template <typename PositionType, std::enable_if_t<not std::is_integral<PositionType>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(strings_column_view&,
                                             column_view&, column_view&,
                                             rmm::mr::device_memory_resource*,
                                             cudaStream_t ) const
    {
        CUDF_FAIL("Positions values for substring must be an integral type.");
    }
};
} // namespace

// Returns strings column of substrings each string in the provided column.
std::unique_ptr<cudf::column> substring_from( strings_column_view strings,
                                              column_view starts_column, column_view stops_column,
                                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                              cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( starts_column.size()==strings_count, "Parameter starts must have the same number of rows as strings.");
    CUDF_EXPECTS( stops_column.size()==strings_count, "Parameter stops must have the same number of rows as strings.");
    CUDF_EXPECTS( starts_column.type()==stops_column.type(), "Parameters starts and stops must be of the same type.");
    CUDF_EXPECTS( starts_column.null_count()==0, "Parameter starts must not contain nulls." );
    CUDF_EXPECTS( stops_column.null_count()==0, "Parameter stops must not contain nulls." );

    // perhaps another candidate for index-normalizer
    return cudf::experimental::type_dispatcher(starts_column.type(),
                dispatch_substring_from_fn{},
                strings, starts_column, stops_column, mr, stream );
}

} // namespace detail

// APIS

std::unique_ptr<cudf::column> substring( strings_column_view strings,
                                         int32_t start, int32_t stop, int32_t step,
                                         rmm::mr::device_memory_resource* mr )
{
    return detail::substring(strings, start, stop, step, mr );
}

//
std::unique_ptr<cudf::column> substring_from( strings_column_view strings,
                                              column_view starts, column_view stops,
                                              rmm::mr::device_memory_resource* mr )
{
    return detail::substring_from(strings, starts, stops, mr );
}

} // namespace strings
} // namespace cudf
