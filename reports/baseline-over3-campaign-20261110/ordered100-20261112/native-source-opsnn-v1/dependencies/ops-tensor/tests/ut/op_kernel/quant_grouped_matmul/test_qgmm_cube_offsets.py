"""Host regression for the actual kernel offset method, not MMAD/Fixpipe emulation.

Run with python3 test_qgmm_cube_offsets.py [--baseline]. The optional baseline
loads the committed header so the sparse regression can demonstrate failure.
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
start = source.index("    __aicore__ inline void UpdateBaseOffsets(")
end = source.index("\n    template <bool isLastGroupAndNeedSplit>", start)
method = source[start:end].replace("__aicore__", "")
harness = r"""
#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>
#include <stdexcept>
namespace asc::te {
using std::get;
template<class T> constexpr int64_t c0_element = 32;
}
constexpr int MNK_M=0, MNK_N=1, MNK_K=2;
constexpr int GMM_CUBE_SPLIT_M=0, GROUP_LIST_TYPE_SPARSE=2, GMM_CUBE_NZ_OUTER_SIZE=16;
int64_t CeilAlign(int64_t x, int64_t a) { return (x+a-1)/a*a; }
template<bool WEIGHT_NZ, bool TRANS_B> struct Kernel {
using BType = int8_t;
std::tuple<int64_t,int64_t,int64_t> problemShape_;
int groupType_=0, groupListType_=2;
int64_t aOffset_=0,wOffset_=0,biasOffset_=0,cOffset_=0,scaleOffset_=0;
int64_t xBaseOffset_=0,wBaseOffset_=0,nAxisBaseOffset_=0,yBaseOffset_=0,mAxisBaseOffset_=0;
METHOD
};
// Uniform values within an expert's padded weight block allow all layouts to
// exercise their physical group stride without duplicating device layout code.
template<bool NZ, bool TRANS> void Run(const std::vector<int>& ids,
    const std::vector<int>& rows, int mode, int feature, int n, int k) {
const int e=ids.size();
const int64_t stride=NZ ? (TRANS ? ((n+15)/16*16)*((k+31)/32*32)
                                      : ((n+31)/32*32)*((k+15)/16*16)) : n*k;
int total=0; for(int m:rows) total+=m;
std::vector<int64_t> x(total*k), w(e*stride), bias(e*n), scale(e*n), y(total*n,-1);
for(int r=0;r<total;++r) for(int j=0;j<k;++j) x[r*k+j]=1+r%3;
for(int g=0;g<e;++g) {
  for(int64_t j=0;j<stride;++j) w[g*stride+j]=feature==0 ? g+1 : 1;
  for(int j=0;j<n;++j) { bias[g*n+j]=feature==1 ? g+2 : 0; scale[g*n+j]=feature==2 ? g+1 : 1; }
}
// Each simulated core advances through all groups, but writes only its rows.
for(int core=0;core<4;++core) {
 Kernel<NZ,TRANS> op; op.groupListType_=mode;
 for(int i=0;i<e;++i) {
  op.problemShape_={rows[i],n,k}; op.UpdateBaseOffsets(ids[i]);
  if(rows[i]==0 && mode==2) break;
  for(int r=0;r<rows[i];++r) {
   if((op.cOffset_/n+r)%4!=core) continue;
   for(int c=0;c<n;++c) {
    int64_t acc=0;
    for(int j=0;j<k;++j) acc+=x.at(op.aOffset_+r*k+j)*w.at(op.wOffset_);
    y.at(op.cOffset_+r*n+c)=(acc+bias.at(op.biasOffset_+c))*scale.at(op.scaleOffset_+c);
   }
  }
 }
}
int row=0;
for(int i=0;i<e;++i) for(int r=0;r<rows[i];++r,++row) for(int c=0;c<n;++c) {
 int g=ids[i]; int64_t expected=(k*(1+row%3)*(feature==0?g+1:1)+(feature==1?g+2:0))*(feature==2?g+1:1);
 if(y[row*n+c]!=expected) throw std::runtime_error("expert/row numerical mismatch");
}
}
template<bool NZ,bool TRANS> void Suite() {
for(int feature=0;feature<3;++feature) for(auto shape: {std::array<int,2>{64,64}, {35,33}}) {
 Run<NZ,TRANS>({0,2,3,1},{64,128,64,0},2,feature,shape[0],shape[1]);
 Run<NZ,TRANS>({2,0,1},{64,32,0},2,feature,shape[0],shape[1]);
 Run<NZ,TRANS>({2,0,1},{1,1,0},2,feature,shape[0],shape[1]);
 for(int mode=0;mode<2;++mode) Run<NZ,TRANS>({0,1,2},{1,0,3},mode,feature,shape[0],shape[1]);
}
}
int main() {
try { Suite<false,false>(); Suite<false,true>(); Suite<true,false>(); Suite<true,true>(); }
catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
std::cout<<"120 offset-driven numerical cases passed (not device MMAD/Fixpipe validation)\n";
}
""".replace("METHOD", method)
with tempfile.TemporaryDirectory(prefix="qgmm-offsets-") as directory:
    cpp = pathlib.Path(directory) / "test.cpp"
    binary = pathlib.Path(directory) / "test"
    cpp.write_text(harness)
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run([str(binary)])
    sys.exit(result.returncode)
