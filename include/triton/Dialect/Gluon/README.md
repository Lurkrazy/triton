# Gluon IR: Automatic Layout Inference for GPU Computing

Gluon IR is an experimental intermediate representation and dialect within the Triton compiler stack that provides automatic layout inference for GPU kernel development. It simplifies GPU programming by automatically determining optimal memory layouts for tensors, reducing the need for manual layout specification while maintaining high performance.

## Overview

The Gluon dialect extends Triton's MLIR-based infrastructure with:

- **Automatic Encoding Inference**: Automatically infers memory layouts for tensors based on usage patterns
- **Higher-Level Programming Model**: Provides abstractions that reduce boilerplate code 
- **Multi-Architecture Support**: Targets NVIDIA and AMD GPU architectures
- **Layout Optimization**: Automatically optimizes data layout decisions for performance
- **Seamless Integration**: Works within the existing Triton compilation pipeline

## Core Concepts

### Auto-Encoding

The central concept in Gluon IR is the `AutoEncodingAttr` - a special encoding that represents "to-be-determined" layouts. These are automatically resolved during compilation through layout inference:

```mlir
// Tensor with automatic encoding - layout will be inferred
%tensor = ... : tensor<128xf32, #gluon.auto_encoding>

// Conversion to concrete layout
%concrete = gluon.set_auto_layout %tensor : tensor<128xf32, #gluon.auto_encoding> 
                                          -> tensor<128xf32, #ttg.blocked<{...}>>
```

### Layout Inference

The Gluon dialect implements sophisticated layout inference through the `GluonInferLayoutInterface`, which propagates layout information bidirectionally through the computation graph. The inference process:

1. **Seeds** the graph with concrete layouts from `set_auto_layout` operations
2. **Propagates** layouts through operations using inference rules
3. **Resolves** conflicts when multiple layouts are possible
4. **Optimizes** for performance characteristics

## Architecture

### IR Components

#### Dialect Definition (`GluonDialect.td`)
```tablegen
def Gluon_Dialect : Dialect {
  let name = "gluon";
  let cppNamespace = "::mlir::triton::gluon";
  let description = [{
    Gluon dialect for automatic layout inference.
  }];
}
```

#### Attributes (`GluonAttrDefs.td`)
- **`AutoEncodingAttr`**: Represents automatic layout inference points
  ```tablegen
  def Gluon_AutoEncodingAttr : AttrDef<Gluon_Dialect, "AutoEncoding"> {
    let mnemonic = "auto_encoding";
    let description = [{
      An encoding that is inferred from neighboring ops in the graph.
    }];
  }
  ```

#### Operations (`GluonOps.td`)
- **`SetAutoLayoutOp`**: Converts from auto-encoding to concrete layouts
  ```tablegen
  def Gluon_SetAutoLayoutOp : Gluon_Op<"set_auto_layout"> {
    let summary = "set auto encoding to a concrete encoding type";
    let arguments = (ins TT_Tensor:$src);
    let results = (outs TT_Tensor:$result);
  }
  ```

### File Organization

```
include/triton/Dialect/Gluon/
├── IR/
│   ├── GluonDialect.td      # Dialect definition
│   ├── GluonOps.td          # Operation definitions  
│   ├── GluonAttrDefs.td     # Attribute definitions
│   └── Dialect.h            # C++ headers
├── Transforms/
│   ├── Passes.td            # Pass definitions
│   └── Passes.h             # Pass headers

lib/Dialect/Gluon/
├── IR/
│   └── Dialect.cpp          # IR implementation
└── Transforms/
    ├── ResolveAutoEncodings.cpp  # Layout inference pass
    ├── Canonicalize.cpp          # Canonicalization pass
    └── Inline.cpp                # Inlining pass

python/
├── src/gluon_ir.cc          # Python bindings
└── triton/experimental/gluon/
    ├── language/            # High-level language API
    ├── _runtime.py          # JIT compilation
    └── _compiler.py         # Compiler interface
```

## Transformation Passes

### 1. Resolve Auto Encodings (`gluon-resolve-auto-encodings`)

The core pass that performs automatic layout inference:

**Algorithm Overview:**
1. **Seed Collection**: Finds `set_auto_layout` operations that provide concrete encodings
2. **Bidirectional Propagation**: Spreads layout information through data dependencies
3. **Conflict Resolution**: Handles cases where multiple layouts are possible
4. **Fixed-Point Iteration**: Continues until all auto-encodings are resolved

**Key Features:**
- Handles control flow constructs (`scf.for`, `scf.while`, `scf.if`)
- Supports "may vary" encodings for flexible operations
- Uses hash-based memoization for performance
- Provides detailed debugging output

### 2. Canonicalization (`gluon-canonicalize`)

A reduced set of canonicalization patterns tailored for Gluon IR:

```cpp
// Selected Triton patterns that preserve layout information
LoadOp::getCanonicalizationPatterns(patterns, ctx);
StoreOp::getCanonicalizationPatterns(patterns, ctx); 
BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
// Notably excludes ConvertLayoutOp patterns
```

### 3. Inlining (`gluon-inline`)

Aggressive function inlining to expose layout inference opportunities:

```cpp
pm.addPass(createInlinerPass(/*opPipelines=*/{}, [](OpPassManager &pm) {
  pm.addPass(gluon::createGluonCanonicalize());
}));
```

## Python API

### High-Level Language Interface

The `triton.experimental.gluon.language` module provides a high-level programming interface:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def example_kernel(input_ptr, output_ptr, n_elements: ttgl.constexpr):
    # Automatic layout inference - no manual layout specification needed
    pid = ttgl.program_id(0)
    offsets = pid * 256 + ttgl.arange(0, 256)
    mask = offsets < n_elements
    
    # Layouts are automatically inferred
    data = ttgl.load(input_ptr + offsets, mask=mask)
    result = data * 2.0
    ttgl.store(output_ptr + offsets, result, mask=mask)
```

### Layout Classes

Explicit layouts are available when needed:

```python
# Blocked layout specification
layout = ttgl.BlockedLayout(
    size_per_thread=[4], 
    threads_per_warp=[32], 
    warps_per_cta=[4], 
    order=[0]
)

# Auto layout (default)
auto_layout = ttgl.AutoLayout()

# Architecture-specific layouts
shared_layout = ttgl.SwizzledSharedLayout(vec=4, perPhase=2, maxPhase=8, order=[0])
```

### Compilation Pipeline

```python
# JIT compilation with automatic layout inference
@gluon.jit
def kernel(...):
    # Kernel implementation using auto layouts
    pass

# Compilation happens automatically on first call
# Auto-encodings are resolved during compilation
```

## Usage Examples

### Basic Tensor Operations

```python
@gluon.jit  
def elementwise_add(a_ptr, b_ptr, output_ptr, n_elements: ttgl.constexpr):
    pid = ttgl.program_id(0)
    block_start = pid * 256
    offsets = block_start + ttgl.arange(0, 256)
    mask = offsets < n_elements
    
    # Automatic layout inference for all tensors
    a = ttgl.load(a_ptr + offsets, mask=mask)
    b = ttgl.load(b_ptr + offsets, mask=mask) 
    c = a + b
    ttgl.store(output_ptr + offsets, c, mask=mask)
```

### Matrix Operations

```python
@gluon.jit
def matrix_multiply(a_ptr, b_ptr, c_ptr, 
                   M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr):
    # Layout inference handles matrix multiply layouts automatically
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    
    # Block tiling
    offs_am = pid_m * 64 + ttgl.arange(0, 64)[:, None]
    offs_bn = pid_n * 64 + ttgl.arange(0, 64)[None, :]
    offs_k = ttgl.arange(0, 64)
    
    # Accumulator with inferred layout
    accumulator = ttgl.zeros((64, 64), dtype=ttgl.float32)
    
    for k in range(0, K, 64):
        a = ttgl.load(a_ptr + offs_am * K + (k + offs_k))
        b = ttgl.load(b_ptr + (k + offs_k[:, None]) * N + offs_bn)
        accumulator += ttgl.dot(a, b)
    
    ttgl.store(c_ptr + offs_am * N + offs_bn, accumulator)
```

### Mixed Auto and Explicit Layouts

```python
@gluon.jit
def mixed_layouts(input_ptr, output_ptr, n_elements: ttgl.constexpr):
    # Start with auto layout
    data = ttgl.load(input_ptr + offsets)  # Auto layout inferred
    
    # Convert to specific layout if needed for performance
    blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0]
    )
    data_blocked = ttgl.convert_layout(data, blocked_layout)
    
    # Continue with auto layout inference
    result = data_blocked * 2.0  # Layout propagated automatically
    ttgl.store(output_ptr + offsets, result)
```

## Architecture-Specific Features

### NVIDIA GPU Support
- **Ampere**: Async copy operations, MMA instructions
- **Hopper**: TMA (Tensor Memory Accelerator), advanced barriers
- **Blackwell**: Tensor memory, TCGEN05 operations

### AMD GPU Support  
- **CDNA3/CDNA4**: MFMA instructions, async copy operations
- **Architecture-specific layouts**: AMDMFMA encodings

## IR Generation and Lowering

### Python to Gluon IR
```python
# Python code with @gluon.jit
@gluon.jit
def kernel(x, y):
    result = x + y
    return result
```

### Generated Gluon IR
```mlir
func.func @kernel(%arg0: tensor<128xf32, #gluon.auto_encoding>, 
                  %arg1: tensor<128xf32, #gluon.auto_encoding>) -> tensor<128xf32, #gluon.auto_encoding> {
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32, #gluon.auto_encoding>
  return %0 : tensor<128xf32, #gluon.auto_encoding>
}
```

### After Layout Resolution
```mlir
func.func @kernel(%arg0: tensor<128xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, 
                  %arg1: tensor<128xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) 
                  -> tensor<128xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> {
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
  return %0 : tensor<128xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
}
```

## Testing and Validation

The Gluon IR includes comprehensive testing:

### Unit Tests (`python/test/gluon/`)
- **`test_core.py`**: Basic functionality and layout inference
- **`test_frontend.py`**: Python API and compilation
- **`test_lowerings.py`**: Code generation and optimization
- **`test_consan.py`**: Canonicalization and passes

### FileCheck Tests
```python
@filecheck_test
@gluon.jit
def test_layout_propagation():
    # CHECK: tensor<128xf32, #ttg.blocked<{{.*}}>>
    data = ttgl.arange(0, 128)
    result = data * 2
    return result
```

## Debugging and Development

### Debug Output
Enable detailed layout inference logging:
```bash
export MLIR_ENABLE_DEBUGGING=1
# Run with debug output for gluon-resolve-auto-encodings pass
```

### Layout Inspection
```python
# Check inferred layouts during development
def inspect_layout(tensor):
    print(f"Tensor layout: {tensor.type.encoding}")
```

### Common Issues and Solutions

1. **Unresolved Auto-Encodings**: Ensure `set_auto_layout` operations provide sufficient concrete layout hints
2. **Layout Conflicts**: Use "may vary" encodings for flexible operations  
3. **Performance Issues**: Profile layout choices and use explicit layouts for critical paths

## Future Directions

- **Enhanced Inference**: More sophisticated layout optimization strategies
- **Dynamic Layouts**: Runtime layout adaptation based on data characteristics
- **Cross-Architecture Optimization**: Portable layout strategies across GPU vendors
- **Integration**: Deeper integration with Triton's existing optimization passes

## Contributing

When extending Gluon IR:

1. **Add Operations**: Define new operations in `GluonOps.td`
2. **Extend Inference**: Add inference rules to `GluonInferLayoutInterface`
3. **Update Python API**: Add high-level constructs to the language module
4. **Add Tests**: Include both unit tests and FileCheck tests
5. **Document Changes**: Update this README and inline documentation

The Gluon IR represents a significant step forward in making GPU programming more accessible while maintaining the performance characteristics that make Triton attractive for high-performance computing applications.