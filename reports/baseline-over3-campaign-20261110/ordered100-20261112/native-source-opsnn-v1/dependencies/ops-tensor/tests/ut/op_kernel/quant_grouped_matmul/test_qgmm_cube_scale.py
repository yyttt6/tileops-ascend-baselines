"""Exercise the actual scalar-scale method on host (not device Fixpipe).

Run with --baseline to demonstrate the committed bf16 doubleScale regression.
"""

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
HEADER = "include/blaze/gemm/kernel/kernel_qgmm_cube.h"
source = (
    subprocess.check_output(["git", "show", "HEAD:" + HEADER], cwd=ROOT, text=True)
    if "--baseline" in sys.argv
    else (ROOT / HEADER).read_text()
)
start = source.index("    __aicore__ inline void UpdateScaleScalar(")
end = source.index("    __aicore__ inline void SetMNK(", start)
method = (
    source[start:end]
    .removesuffix("    template <class GroupListTensor>\n")
    .replace("__aicore__", "")
    .replace("__gm__", "")
)
harness = r"""
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
struct bfloat16_t { uint16_t bits; };
namespace AscendC {
template<class A, class B> using IsSameType = std::is_same<A,B>;
template<class T> struct GlobalTensor {
 T* ptr;
 void SetGlobalBuffer(T* p) { ptr=p; }
 T GetValue(int i) { return ptr[i]; }
};
}
enum class QuantMode { DEFAULT, PERTENSOR_MODE, PERCHANNEL_MODE };
constexpr uint32_t LEFT_SHIFT_16=16;
constexpr uint64_t DEQ_SCALE_MUL=0xFFFFE000;
template<class S, class C=float> struct Kernel {
 using ScaleGmType=S; using CType=C;
 bool isPerChannel_=false;
 int x1QuantMode_=1, x2QuantMode_=1;
 S* scaleBBasePtr_=nullptr;
 float* pertokenScaleBasePtr_=nullptr;
 uint64_t scaleScalar_=0;
METHOD
};
int main() {
 // Distinct adjacent values catch both reinterpretation and incorrect stride.
 bfloat16_t bf[]={{0x3fc0},{0x4020},{0x4080},{0x4100}};
 float fp[]={1.5f,2.5f,4.f,8.f}, a[]={2.f,0.5f,0.25f,0.125f};
 for(int mode: {0,1}) for(int i=0;i<4;++i) {
  Kernel<bfloat16_t> b; b.scaleBBasePtr_=bf; b.pertokenScaleBasePtr_=a; b.x1QuantMode_=mode;
  Kernel<float> f; f.scaleBBasePtr_=fp; f.pertokenScaleBasePtr_=a; f.x1QuantMode_=mode;
  b.UpdateScaleScalar(i); f.UpdateScaleScalar(i);
  float expected=fp[i]*(mode ? a[i] : 1.f); uint32_t bits;
  std::memcpy(&bits,&expected,sizeof(bits));
  if(b.scaleScalar_!=(bits&DEQ_SCALE_MUL) || f.scaleScalar_!=(bits&DEQ_SCALE_MUL)) {
   std::cerr<<"scale mismatch: mode="<<mode<<" group="<<i<<'\n'; return 1;
  }
 }
 Kernel<bfloat16_t,int32_t> raw; raw.UpdateScaleScalar(0);
 if(raw.scaleScalar_!=0) return 1;
 std::cout<<"16 bf16/fp32 scalar checks and INT32 no-scale check passed\n";
}
""".replace("METHOD", method)
with tempfile.TemporaryDirectory(prefix="qgmm-scale-") as directory:
    cpp = pathlib.Path(directory) / "test.cpp"
    binary = pathlib.Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O0",
            "-fno-strict-aliasing",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    sys.exit(subprocess.run([str(binary)]).returncode)
