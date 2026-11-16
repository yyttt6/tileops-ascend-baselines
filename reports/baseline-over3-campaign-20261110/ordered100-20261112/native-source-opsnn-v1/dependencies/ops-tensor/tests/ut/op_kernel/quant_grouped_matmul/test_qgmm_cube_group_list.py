"""Host regression of the actual group-list parsing method with checked reads."""

import pathlib
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[4]
s = (root / "include/blaze/gemm/kernel/kernel_qgmm_cube.h").read_text()
a = s.index(
    "    template <class GroupListTensor>\n    __aicore__ inline int64_t GetSplitValue"
)
b = s.index("    __aicore__ inline void UpdateBaseOffsets", a)
method = s[a:b].replace("__aicore__", "")
harness = r"""
#include <cstdint>
#include <vector>
#include <cassert>
constexpr int GMM_CUBE_NO_SPLIT=-1, GROUP_LIST_TYPE_OFFSET=0, GROUP_LIST_TYPE_LENGTH=1;
constexpr int SPARSE_GROUP_LIST_ITEM_STRIDE=2, SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET=1;
struct List {
 std::vector<int64_t> data;
 int64_t operator[](size_t i) const { return data.at(i); }
};
struct Kernel { int groupType_=0, groupListType_=0; int64_t preOffset_=0;
METHOD
};
int main() {
 for(int mode=0;mode<3;++mode) {
  Kernel k; k.groupType_=-1; k.groupListType_=mode;
  assert(k.GetSplitValueFromGroupList(List{},7)==0);
 }
 for(int mode=0;mode<3;++mode) {
  Kernel k; k.groupListType_=mode;
  List list{mode==0 ? std::vector<int64_t>{2,2,5} : mode==1 ? std::vector<int64_t>{2,0,3} : std::vector<int64_t>{2,2,0,0,1,3}};
  assert(k.GetSplitValueFromGroupList(list,0)==2);
  assert(k.GetSplitValueFromGroupList(list,1)==0);
  assert(k.GetSplitValueFromGroupList(list,2)==3);
 }
}
""".replace("METHOD", method).replace(
    "#include <cstdint>", "#include <cstdint>\n#include <cstddef>"
)
with tempfile.TemporaryDirectory() as d:
    cpp = pathlib.Path(d) / "test.cpp"
    cpp.write_text(harness)
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", d + "/test"], check=True)
    subprocess.run([d + "/test"], check=True)
print("NO_SPLIT no-read and OFFSET/LENGTH/SPARSE parsing passed")
