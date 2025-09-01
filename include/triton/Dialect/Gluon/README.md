# Gluon IR: Automatic Layout Inference for GPU Computing

Gluon IR is an experimental intermediate representation and dialect within the Triton compiler stack that provides automatic layout inference for GPU kernel development. It simplifies GPU programming by automatically determining optimal memory layouts for tensors, reducing the need for manual layout specification while maintaining high performance.

## Overview

The Gluon dialect extends Triton's MLIR-based infrastructure with:

- **Automatic Encoding Inference**: Automatically infers memory layouts for tensors based on usage patterns
- **Higher-Level Programming Model**: Provides abstractions that reduce boilerplate code 
- **Multi-Architecture Support**: Targets NVIDIA and AMD GPU architectures
- **Layout Optimization**: Automatically optimizes data layout decisions for performance
- **Seamless Integration**: Works within the existing Triton compilation pipeline

## Triton Frontend vs. Gluon IR: Key Differences

Understanding when to use the traditional Triton frontend versus Gluon IR is crucial for optimal GPU kernel development. Both approaches expose GPU hardware features and provide developer control, but they differ significantly in their programming models and abstraction levels.

### Programming Model Comparison

#### Traditional Triton Frontend
The standard Triton frontend (`triton.language`) requires explicit specification of tensor layouts and memory encodings:

```python
import triton
import triton.language as tl

@triton.jit
def traditional_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Explicit layout - must specify encoding details
    input_data = tl.load(input_ptr + offsets, mask=mask)
    result = input_data * 2.0
    tl.store(output_ptr + offsets, result, mask=mask)
```

**Layout Specification Required:**
```python
# Manual layout configuration in traditional Triton
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    # Explicit tensor shaping and layout management
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Manual accumulator initialization with explicit shape
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Manual offset calculations and layout considerations
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        
        offs_k += BLOCK_SIZE_K
```

#### Gluon IR Frontend  
Gluon IR (`triton.experimental.gluon`) provides automatic layout inference:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def gluon_kernel(input_ptr, output_ptr, n_elements: ttgl.constexpr):
    pid = ttgl.program_id(0)
    offsets = pid * 256 + ttgl.arange(0, 256)
    mask = offsets < n_elements
    
    # Automatic layout inference - no explicit encoding needed
    input_data = ttgl.load(input_ptr + offsets, mask=mask)
    result = input_data * 2.0
    ttgl.store(output_ptr + offsets, result, mask=mask)
```

**Automatic Layout Inference:**
```python
# Automatic layout management in Gluon IR
@gluon.jit
def gluon_matmul(a_ptr, b_ptr, c_ptr, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    
    # Simplified offset calculations - layouts inferred automatically
    offs_am = pid_m * 64 + ttgl.arange(0, 64)[:, None]
    offs_bn = pid_n * 64 + ttgl.arange(0, 64)[None, :]
    offs_k = ttgl.arange(0, 64)
    
    # Automatic accumulator layout optimization
    accumulator = ttgl.zeros((64, 64), dtype=ttgl.float32)
    
    for k in range(0, K, 64):
        # Automatic layout propagation through loads and compute
        a = ttgl.load(a_ptr + offs_am * K + (k + offs_k))
        b = ttgl.load(b_ptr + (k + offs_k[:, None]) * N + offs_bn)
        accumulator += ttgl.dot(a, b)  # Layout automatically optimized for dot product
    
    ttgl.store(c_ptr + offs_am * N + offs_bn, accumulator)
```

### GPU Hardware Feature Exposure

Both frontends expose GPU hardware features, but at different abstraction levels:

#### Traditional Triton: Direct Hardware Control
```python
# Explicit shared memory layout control
@triton.jit
def traditional_shared_memory(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    # Manual shared memory allocation and layout specification
    shared_mem = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Explicit memory coalescing patterns
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(input_ptr + tid)
    
    # Manual synchronization and shared memory usage
    tl.store(shared_mem_ptr, data)
    # ... explicit barrier synchronization needed
```

**Hardware-Specific Features:**
- Direct control over shared memory layouts
- Manual specification of memory coalescing patterns
- Explicit tensor memory accelerator (TMA) usage on Hopper
- Direct control over warp-level primitives
- Manual async copy operation management

#### Gluon IR: High-Level Hardware Optimization
```python
# Automatic hardware feature utilization
@gluon.jit  
def gluon_optimized(input_ptr, output_ptr, n_elements: ttgl.constexpr):
    # Hardware features automatically selected based on target architecture
    data = ttgl.load(input_ptr + offsets)  # Automatic TMA usage on Hopper if beneficial
    
    # Shared memory usage automatically optimized
    result = ttgl.dot(data, weights)  # MMA/WMMA instructions automatically selected
    
    ttgl.store(output_ptr + offsets, result)  # Optimal memory coalescing automatically applied
```

**Automatic Hardware Optimization:**
- Intelligent selection of memory access patterns (TMA vs. traditional loads)
- Automatic MMA/WMMA instruction utilization based on data layouts
- Smart shared memory usage without manual management
- Architecture-specific optimization (Ampere vs. Hopper vs. Blackwell)
- Automatic async copy optimization

### Developer Control Comparison

#### Traditional Triton: Explicit Control
**Advantages:**
- Complete control over memory layouts and access patterns
- Direct specification of hardware feature usage
- Predictable performance characteristics
- Fine-tuned optimization for specific use cases

**Challenges:**
- Requires deep understanding of GPU architecture
- Verbose code with manual layout management
- Error-prone memory access pattern specification
- Difficult to port across different GPU architectures

```python
# Explicit layout control in traditional Triton
@triton.jit
def explicit_control_example(a_ptr, b_ptr, output_ptr, 
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Manual specification of blocked layout
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Explicit memory access pattern design
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    
    # Manual specification of tensor shapes and strides
    a_ptrs = a_ptr + offs_am  # Requires careful stride calculation
    b_ptrs = b_ptr + offs_bn  # Manual memory coalescing consideration
    
    # Developer responsible for optimal memory access patterns
    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)
```

#### Gluon IR: Intelligent Automation with Override Capability
**Advantages:**
- Automatic optimization while preserving control when needed
- Simplified development workflow
- Portable across GPU architectures
- Reduced development time and bugs

**Flexibility:**
- Can override automatic decisions with explicit layouts when needed
- Provides both high-level abstractions and low-level access
- Automatic fallback to traditional Triton for unsupported patterns

```python
# Intelligent automation with override capability
@gluon.jit
def flexible_control_example(input_ptr, output_ptr, use_custom_layout: ttgl.constexpr):
    data = ttgl.load(input_ptr + offsets)  # Automatic layout inference
    
    if use_custom_layout:
        # Override automatic layout when specific control is needed
        custom_layout = ttgl.BlockedLayout(
            size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0]
        )
        data = ttgl.convert_layout(data, custom_layout)
    
    # Continue with automatic optimization
    result = data * 2.0  # Layout automatically propagated
    ttgl.store(output_ptr + offsets, result)
```

### Performance and Use Case Guidelines

#### When to Use Traditional Triton
**Best For:**
- Performance-critical kernels requiring fine-tuned control
- Specific hardware feature exploitation (e.g., specialized TMA patterns)
- Kernels with unusual memory access patterns
- Research into new GPU programming techniques
- Maximum performance extraction from specific architectures

**Examples:**
- Custom matrix multiplication with non-standard blocking
- Specialized reduction algorithms with custom shared memory usage
- Kernels exploiting specific Hopper/Blackwell features
- Research kernels exploring new algorithmic approaches

#### When to Use Gluon IR
**Best For:**
- Rapid prototyping and development
- Portable kernels across GPU architectures  
- Standard computational patterns (element-wise, reductions, GEMM)
- Production kernels where development velocity is important
- Kernels that need to adapt to different hardware generations

**Examples:**
- Element-wise tensor operations
- Standard convolution implementations
- Transformer attention mechanisms
- Activation functions and normalization layers
- General-purpose computational kernels

### Architecture-Specific Benefits

#### NVIDIA GPU Features
**Traditional Triton:**
```python
# Explicit Hopper TMA usage
@triton.jit
def explicit_tma_kernel(...):
    # Manual TMA barrier and copy operations
    tl.experimental_tensormap_fenceproxy_acquire(...)
    # Direct control over TMA descriptor usage
```

**Gluon IR:**
```python
# Automatic TMA optimization
@gluon.jit
def auto_tma_kernel(...):
    # TMA automatically used when beneficial for Hopper+ targets
    data = ttgl.load(...)  # Automatic TMA utilization
```

#### AMD GPU Features
**Traditional Triton:**
```python
# Explicit MFMA instruction usage
@triton.jit  
def explicit_mfma(...):
    # Manual MFMA layout and instruction selection
    # Requires AMD-specific layout knowledge
```

**Gluon IR:**
```python
# Automatic MFMA optimization
@gluon.jit
def auto_mfma(...):
    # MFMA automatically selected for matrix operations on CDNA architectures
    result = ttgl.dot(a, b)  # Optimal MFMA usage automatically applied
```

### Migration Path

For teams transitioning between approaches:

1. **Start with Gluon IR** for rapid development and prototyping
2. **Profile and identify bottlenecks** using automatic optimization
3. **Selectively use traditional Triton** for performance-critical sections
4. **Leverage hybrid approaches** where Gluon handles standard patterns and traditional Triton handles specialized optimizations

```python
# Hybrid approach example
@gluon.jit
def hybrid_kernel(input_ptr, output_ptr, specialized_section: ttgl.constexpr):
    # Standard operations use Gluon automatic optimization
    data = ttgl.load(input_ptr + offsets)
    
    if specialized_section:
        # Drop to traditional Triton for specialized control
        # (Implementation would require careful interface design)
        result = custom_triton_operation(data)
    else:
        # Continue with automatic optimization
        result = data * 2.0
    
    ttgl.store(output_ptr + offsets, result)
```

This comparison demonstrates that both Triton frontend and Gluon IR provide comprehensive GPU hardware access, but Gluon IR adds an intelligent automation layer that reduces complexity while preserving the ability to override automatic decisions when maximum control is needed.

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