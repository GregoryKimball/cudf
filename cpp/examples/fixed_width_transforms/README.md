# libcudf fixed-width transform examples

## Oracle `NUMBER`-inspired decimal with LTO transform

`oracle_number_inspired_lto` demonstrates a multi-input, multi-output UDF compiled to a fatbin and
linked at runtime by `cudf::transform_lto`.

Each number is exposed as a struct with two children:

```text
STRUCT<coefficient: DECIMAL128(scale=0), exponent: INT16>
value = coefficient * 10^exponent
```

The representation and arithmetic contract are inspired by Oracle `NUMBER`'s guaranteed 38-digit
precision and documented finite range (`1e-130 <= abs(value) < 1e126`). This example does not claim
full Oracle compatibility: it does not use Oracle's variable-length base-100 storage format or
preserve the optional 39th or 40th physical mantissa digits.

The stored coefficient is 128-bit, but addition and multiplication use a five-limb intermediate
with 95 decimal digits of capacity. This accommodates exact exponent alignment that can influence
a 38-digit sum and the full 76-digit product of two coefficients. Results are rounded to 38
significant digits, ties away from zero, then canonicalized by removing trailing coefficient zeros.
Exact compatibility with Oracle arithmetic at boundary cases is outside the scope of this example.

`transform_lto` does not accept struct inputs or outputs directly. The host wrapper passes the four
fixed-width input children and two output children, then wraps the output children in a struct.

Build and run:

```bash
cmake -S cpp/examples/fixed_width_transforms \
      -B cpp/examples/fixed_width_transforms/build
cmake --build cpp/examples/fixed_width_transforms/build --parallel

cpp/examples/fixed_width_transforms/build/oracle_number_inspired_lto add
cpp/examples/fixed_width_transforms/build/oracle_number_inspired_lto multiply
```
