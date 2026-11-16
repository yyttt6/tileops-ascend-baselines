"""Exact CPU-lazy routes to unchanged, verified whole CausalConv1dUpdate APIs."""
import builtins, hashlib, json, os, types, tempfile
from pathlib import Path
ROOT = '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter'
PINS = {'/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/ALL_CONTRACTS.json': '90586198f68935cfa0aacf8dfe9fcdc1cc54d522fa05a858b1f9b44a985675ef',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/PINS.json': 'bfb1d5625d57172ef4cf594e325d8a06ae982854597551720d6d38b9c5c1f1ee',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/RUNTIME_RECIPE.json': '26716d949e366555594a82d1a5be456f4f6dffc1d2a56335fc0ee1b309f08eec',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/binding.py': 'c7c40980dc2111cca02c9219826a4ca931cc91df9bc1156c1ee9f651c4531063',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/direct_acl.py': '9b9326689ee239b3e45c1b88025ed24e1cd4ca37cf078bd7ad4f139c0e65e331',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_api/lib/libcust_opapi.so': '4c4e8716dcbcf5af89467b1fe794080cde6fdeefa1e418a3571fcc14b1a1fc0a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json': '3ed9945b45384bb2892e3e0c5de6a79994ead165f723ce73ac9626289d21ab55',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/ascend950/causal_conv1d_apt/CausalConv1d_31747f8d97ce736b918decc906c76fc0.json': 'f5628c631636112020323263560a729ed3543e3aa5ac558c6a131b933dcdf786',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/ascend950/causal_conv1d_apt/CausalConv1d_31747f8d97ce736b918decc906c76fc0.o': '01464b044f7d52057734aeee9bdc8f93f63100512ace9fea54f57f80f2cd056b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/ascend950/causal_conv1d_apt/CausalConv1d_9ee94e99ad2bb70235a06c1c10559bf4.json': 'a8419906bc2ac2a53e5204ddd0ae9658420063a7541ccb1f0ddfd5a109cc1e47',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/ascend950/causal_conv1d_apt/CausalConv1d_9ee94e99ad2bb70235a06c1c10559bf4.o': 'b6457dab2033fad9d40daf5d08abdfd28b29a91aaf9d6c1b77d8f4d995d2ef9a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json': 'a75c7f5a7e00f7194d20f00eb895b7239a585632c3e1bb86e5f7296169f2c4d2',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/kernel/config/ascend950/causal_conv1d_apt.json': '1de51da9c424f1981fb4061e36807d02bcc41914e8b5f7a67555a62fb539188e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so': 'd6d439827cd647f5520c926f2294c953abfa0b28db2d70fb0b3d2760f46da89e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/op_tiling/liboptiling.so': 'd6d439827cd647f5520c926f2294c953abfa0b28db2d70fb0b3d2760f46da89e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_proto/es/lib/linux/aarch64/libes_transformer_cust.so': 'f19e9889dac51ba20abc7f46ae3f3fd784305e01ba0dccdf1e27437a145fa662',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so': 'e784d2159b337f8cf0036811387b535409f85f8a1629fb8d372b435e780758cf',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/version.info': 'fdd4d0d83c19ab6cd3f874c61c773a54a29e64ac85cd7783f409c4fc4d06c311',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/CMakeLists.txt': 'ad7629516920636e4bbda8370fd9b1aadfd6d4e461744399fea94a687eda39f2',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/build.sh': 'e77d9ad4def9da7961dc619274aa549c7875d04292cc576073518b9b007e06d3',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/cmake/custom_build.cmake': '54cb3a627ea6973b537e88ed838ac680fc1a49469bae30328b829591a3474b45',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/cmake/func.cmake': '6739ceda276942e6de920615530c66ae3186a61da90d5a5dc0d4e8c87d6902cc',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/cmake/scripts/util/ascendc_bin_param_build.py': 'dfeb197493e6c9bae4c26377d25ab027b55898b113da76e7d0a58c010521d4f4',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/cmake/scripts/util/ascendc_ops_config.py': 'b76a543f5bac47a50bdf6f9ec3726ceb404d94d28ce81dfcdac2d20240cc3111',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/CMakeLists.txt': 'faffe61ec4c0aff091cd25eb100680a42121d59835cb3e27da361f5231cbb221',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/README.md': '5bb608dd244e1284b7468a0315412b2f9bc1de97995011755ea3d3af9f9a3482',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/docs/aclnnCausalConv1dFn.md': 'a222132aa92ccb475c45baa097eafc0d81a59600a34a4841f1a0d7b1bdfbdb1a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/docs/aclnnCausalConv1dUpdate.md': '48374a27d3e054666107d77e6b368301e2a3bc8277770dbdadface9c70b988af',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/docs/torchapi_causal_conv1d_fn.md': 'e5c0ef60d3590a7480fbad3fe47888c892b3d71dba58f974bce08e561aba1054',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/docs/torchapi_causal_conv1d_update.md': 'ca0214b73ad37f76455056acb9bbdc874fd22692800b49c474ee887487a3522d',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/examples/test_aclnn_causal_conv1d_fn.cpp': '49468710e9e8b167ef922809a7b59ef61233f21ac0263c2cf8f40f75d5a69e9a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/examples/test_aclnn_causal_conv1d_update.cpp': '81911a7ef9031b67160eed40e627f01c340e738410b6cbb1c856d89f8468dba5',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/examples/test_geir_causal_conv1d.cpp': '00b454603cbd6abbdc3202c053c93dc6ea3669cbe069e225badcbb224ab0abc5',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/aclnn_causal_conv1d_fn.cpp': '9521141ec72dfe687c4b25f93472b5e5c470686428c76b365ae8688d215792e8',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/aclnn_causal_conv1d_fn.h': '97c9be0af28ccdfe31fa5d40fce99aa94e18c3b28a43b634d7c9acae81f65c26',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/aclnn_causal_conv1d_update.cpp': '37bcd5a75f107c9887855cc1f4fe07ef0456fe2605fa61f0a8328275854ae8f1',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/aclnn_causal_conv1d_update.h': '960573b28f0af4c12de4b1ca44daa72d70c1e7c9ff740131489618fcf88a8391',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/causal_conv1d.cpp': '6b0faf761daa65f1244fe9c29188a6d5177921646a73252c72d92a35edd48a28',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_api/causal_conv1d.h': '5b868707b2489f3e39dec4bd7e88c636085dd62fc910ab0cc290cd972383e87f',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_graph/CMakeLists.txt': 'fabc053112e412e7ff0ea80c2bdce9ea71cf5c5617744c9012ab6c71c974ce9e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_graph/causal_conv1d_proto.h': '6fc5a2ff57035a96f9396c0ecfa98d3b04525da2d956598ceb4a13253205ae69',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/CMakeLists.txt': '3fda52734f988cccb2827e5f2e07e84cf0f297939fcbe79f529fbab5661d2a0e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/causal_conv1d_def.cpp': 'f27618be94897a92978fc86c9d0083bb6eec5f07bc610e9bb91e2d1eeb43b282',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/causal_conv1d_infershape.cpp': 'c32797ebb0522bba6c9c03c8c527230d5860969f2953299bf429e8d81a0f7dcb',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/causal_conv1d_tiling.cpp': '6743d319d62590cd16d87a8f8c80592d6ac06bf15ef4ac8198a339e99fe891a7',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/config/ascend350/causal_conv1d_binary.json': '6ee277ced94669da469ba90aeae2b3d9b0277f5709542260d352d7811644761b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/config/ascend350/causal_conv1d_simplified_key.ini': '3f8cee63f890686580c04d8d7abc8e7574232683e00681a60d7b3c7fc28ee6a6',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/config/ascend950/causal_conv1d_binary.json': '6ee277ced94669da469ba90aeae2b3d9b0277f5709542260d352d7811644761b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_host/config/ascend950/causal_conv1d_simplified_key.ini': '36420b48cd705f2bad2ea3ecb452202b9025cc68a9e0a7a711801b0b829a7fac',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d.h': '05d9a2265084a86357777adc221d67a8570e0c75cb5f8bcbded662110d9238d1',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_fn.h': 'a63d194254fab4c00db1f1265a52482517fd094a260f76e09165caf91b2119d3',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_tiling_data.h': 'bb8cb9fb182472190ee7ac91f5ae3bfc64b913bec4505b0a2c8da2369f0cd179',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_tiling_key.h': '00809d5cc3c00a93ca7314f2add8fe856aa95b73d8c25078f2de4ad2ddc783e2',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_update.h': '812584ad8fd5b8dbb76fa53fb15648a15d2942fec7c6d23a1fa866355fea13c3',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/arch35/causal_conv1d_util.h': '368a227ea0fd5449b445784ccb17ce068ab2824fed050c401cbad668f2850404',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/op_kernel/causal_conv1d_apt.cpp': '8e316936041b94261983916eb868d50e3db92e4f7838cb975c6a22abd0b89b0f',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/CMakeLists.txt': '4723f051ab9595a3052c233e141c1f3d1b2fa08e10cd90def70e11c3f7271da1',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/assets/golden.py': 'dde373acdf9f3e69824f8847489af9fde503be68872e977197c81c6169fadaac',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/st/arch35/causal_conv1d_test_cases.csv': 'd167b28efae731ac47b7f4dacb23ceb772d687920f07721db76aa4345a4270f3',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/CMakeLists.txt': '4723f051ab9595a3052c233e141c1f3d1b2fa08e10cd90def70e11c3f7271da1',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/op_host/CMakeLists.txt': '8993dd6b06c8115afce976f8adea3813c718f1abc62a8ff5397a67ca5eba9244',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/op_host/op_api/CMakeLists.txt': 'bfb2e310edb637996cc22217d4020063ba28ca00fc8bf39554230a2bb75b0a7b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/op_host/op_api/test_aclnn_causal_conv1d_fn.cpp': '5b2ae2163be6f096b42cbb70860bc8d7c287a1fe113baa9525f985414d2d7aaa',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/op_host/op_api/test_aclnn_causal_conv1d_update.cpp': '1435b28d0ecce27dd3d07f5bada268db22906bc4e9f3764dc1f4e441142c6f5b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/tests/ut/op_host/test_causal_conv1d_tiling.cpp': 'a243a757c72ef0867931c4f1a8d3317f45aedf069f3c98bba94136cc51110c60',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/__init__.py': 'f37ee9ca0be24073077fa420beda920a2cfd9da56c088b8d5453eea9375fcf60',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/causal_conv1d_fn.py': '346eedce38d90dc0e4b53cfb08cdf98786e853a33507bd09f6da7ee5a0cecec1',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/causal_conv1d_update.py': '5b421124f35b79c241e5f0fef25f1cf3ccc322b8581e4ca780a145548402600a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/csrc/causal_conv1d_fn.cpp': '6b350a8c44acc88d0bc75279a5336efab1efd3b50cc12838cb4aac8fbe19509d',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/csrc/causal_conv1d_update.cpp': 'dfdd322dcf2f93167a2ca94b547eaf03611aeabcc002ecb1116bbd11f393f96b',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/source/mamba/causal_conv1d/torch_extension/graph_convert_causal_conv1d.py': '2605880fdcff30681c68ea8b1a0dc67c9936ae8064485c73cb554db9e0ccfdd2',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libascendcl.so': '123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libnnopbase.so': '308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime.so': '7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_common.so': 'b02fb580c4d3ccebfa0d481f21e5ccea7f135042e8239d57446737decb4f6958',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_v200.so': 'f009d4a33de9a72ff733044857769bc494d1454d83c71f02bce617e2a318f306',
 '/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so': '308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b'}
LOCAL_FILES = ['/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/binding.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/direct_acl.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/PINS.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/adapter/RUNTIME_RECIPE.json',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/ALL_CONTRACTS.json']
LOADED_VENDOR = {'/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_api/lib/libcust_opapi.so': '4c4e8716dcbcf5af89467b1fe794080cde6fdeefa1e418a3571fcc14b1a1fc0a',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so': 'd6d439827cd647f5520c926f2294c953abfa0b28db2d70fb0b3d2760f46da89e',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2/install/vendors/causalpairs_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so': 'e784d2159b337f8cf0036811387b535409f85f8a1629fb8d372b435e780758cf',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libascendcl.so': '123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libnnopbase.so': '308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime.so': '7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_common.so': 'b02fb580c4d3ccebfa0d481f21e5ccea7f135042e8239d57446737decb4f6958',
 '/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_v200.so': 'f009d4a33de9a72ff733044857769bc494d1454d83c71f02bce617e2a318f306'}
CASE_PARAMS = {('ops-transformer', 'CausalConv1dUpdateFwdOp', 'mamba2-2.7b-decode-b1-bfloat16'): '{"activation":"silu","conv_state":[[1,5376,4],"bfloat16"],"output":[[1,5376],"bfloat16"],"weight":[[5376,4],"bfloat16"],"x":[[1,5376],"bfloat16"]}',
 ('ops-transformer', 'CausalConv1dUpdateFwdOp', 'qwen3-next-decode-b64-float16'): '{"activation":"silu","conv_state":[[64,8192,4],"float16"],"output":[[64,8192],"float16"],"weight":[[8192,4],"float16"],"x":[[64,8192],"float16"]}'}
ROOT = Path(ROOT)

def _checked(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('Frozen causal dependency changed: ' + str(path))
    return data

def _snapshot(runtime=False):
    return {p: _checked(p, h) for p, h in PINS.items() if runtime or p in LOCAL_FILES}

def _key(case, baseline='ops-transformer'):
    return baseline, case._call.signature.name, case.id

def matches(case, baseline='ops-transformer'):
    try:
        return json.dumps(case.params, sort_keys=True, separators=(',', ':'), allow_nan=False) == CASE_PARAMS.get(_key(case, baseline))
    except (TypeError, ValueError):
        return False

def _overlay_mapping_matches(path, digest, fields):
    """Compare kernel identities with a fresh read-only mapping of pinned bytes.

    Overlay pathname stat may expose a different device than /proc/maps. No DSO
    is loaded or executed here. Never accept the device mismatch on inode alone.
    """
    import mmap
    before = {line.split()[0] for line in Path('/proc/self/maps').read_text().splitlines()}
    with open(path, 'rb') as source:
        st = os.fstat(source.fileno())
        if st.st_ino != int(fields[4]) or not st.st_size:
            return False
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            if hashlib.sha256(mapping).hexdigest() != digest:
                return False
            identities = set()
            for line in Path('/proc/self/maps').read_text().splitlines():
                entry = line.split(maxsplit=5)
                if (len(entry) == 6 and entry[0] not in before
                    and entry[5] == path and int(entry[2], 16) == 0):
                    identities.add((entry[3], int(entry[4])))
            after = Path(path).stat()
            stable = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            return stable and (fields[3], int(fields[4])) in identities

def _loaded_vendor_pins(require=False):
    expected = LOADED_VENDOR
    names = {Path(p).name for p in expected}; seen = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6: continue
        path = fields[5]
        if Path(path.removesuffix(' (deleted)')).name not in names: continue
        if path.endswith(' (deleted)'): raise RuntimeError('Deleted vendor mapping')
        resolved = str(Path(path).resolve())
        if resolved not in expected: raise RuntimeError('Unexpected vendor mapping: ' + path)
        st = Path(resolved).stat(); major, minor = (int(x, 16) for x in fields[3].split(':'))
        if st.st_ino != int(fields[4]):
            raise RuntimeError('Mapped vendor inode differs from pinned file: ' + path)
        if (os.major(st.st_dev), os.minor(st.st_dev)) != (major, minor):
            if not _overlay_mapping_matches(resolved, expected[resolved], fields):
                raise RuntimeError('Mapped vendor device identity cannot be proved: ' + path)
        _checked(resolved, expected[resolved]); seen.add(resolved)
    if require and seen != set(expected): raise RuntimeError('Missing reviewed loaded vendor libraries')

def _load(data):
    private = {}; original = builtins.__import__
    class FrozenPath(type(Path())):
        def read_bytes(self):
            if str(self) in data: return data[str(self)]
            return super().read_bytes()
        def read_text(self, *args, **kwargs):
            if str(self) in data: return data[str(self)].decode('utf-8')
            return super().read_text(*args, **kwargs)
    pathlib_proxy = types.ModuleType('pathlib'); pathlib_proxy.Path = FrozenPath
    os_proxy = types.ModuleType('os'); os_proxy.__dict__.update(vars(os))
    os_proxy.environ = dict(os.environ)
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'direct_acl' and level == 0: return private['direct_acl']
        if name == 'pathlib' and level == 0: return pathlib_proxy
        if name == 'os' and level == 0: return os_proxy
        return original(name, globals, locals, fromlist, level)
    for name in ('direct_acl', 'binding'):
        path = str(ROOT/(name+'.py')); mod = types.ModuleType('_causal_verified_'+name)
        mod.__file__ = path
        mod.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        exec(compile(data[path], path, 'exec'), mod.__dict__)
        private[name] = mod
    return private

def _require_runtime(private):
    # Original bytes enforce nonempty matching GDN_CASE_OWNER and actual foreground pgrp.
    private['direct_acl'].os.environ = dict(os.environ)
    private['direct_acl'].require_guard()
    vendor = str(ROOT.parent/'install/vendors/causalpairs_transformer')
    if os.environ.get('ASCEND_CUSTOM_OPP_PATH') != vendor:
        raise RuntimeError('Original single server2 vendor environment required')
    home = '/usr/local/Ascend/cann-9.1.0'
    for name, value in {'ASCEND_HOME_PATH':home, 'ASCEND_TOOLKIT_HOME':home,
                        'ASCEND_OPP_PATH':home+'/opp','ASCEND_AICPU_PATH':home,
                        'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','FLA_NPU_DISABLE_PTH':'1'}.items():
        if os.environ.get(name) != value: raise RuntimeError('Original environment required: '+name)

def make(case, baseline='ops-transformer'):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case, baseline): raise UnsupportedCase('Exact causal key and all canonical parameters required')
    _snapshot()  # No runtime libraries loaded by registry construction.
    key = _key(case, baseline)
    frozen = types.SimpleNamespace(id=key[2], params=json.loads(CASE_PARAMS[key]),
        _call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
    child = None; private = None
    def run(*args, **kwargs):
        nonlocal child, private
        if private is None: private = _load(_snapshot())
        _require_runtime(private)  # Always before torch import / allocation / CDLL.
        data = _snapshot(runtime=True)
        _loaded_vendor_pins(require=child is not None)
        # Reject old evidence destinations before native construction.
        root = Path(os.environ['GDN_RUN_DIR']).resolve()
        if root == ROOT.parent or ROOT.parent in root.parents:
            raise RuntimeError('GDN_RUN_DIR must not reuse original build/checker evidence tree')
        if child is None:
            private = _load(data)
            _require_runtime(private)
            child = private['binding'].build(frozen)
        # Each invocation gets a new evidence directory; original binding writes unchanged.
        root.mkdir(parents=True, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix='causal-registration2-', dir=str(root))
        private['binding'].os.environ['GDN_RUN_DIR'] = run_dir
        _loaded_vendor_pins()
        result = child.implementation.run(*args, **kwargs)
        _loaded_vendor_pins(require=True)
        return result
    def cleanup():
        if child is not None:
            _require_runtime(private)
            if child.cleanup is not None: child.cleanup()
    impl = types.SimpleNamespace(run=run, args=None, reset=None, noncomparable_reason=None)
    return Candidate(impl, 'ops-transformer::aclnnCausalConv1dUpdate',
        'Unchanged verified two-stage Update API; full state4, original roll/transpose/copy mapping.',
        dict(PINS), limitations=('Server2 private vendor, original environment and foreground guard required.',),
        timing_mode='host_roundtrip', cleanup=cleanup)

def wrap(previous):
    def dispatch(case):
        if _key(case) in CASE_PARAMS: return make(case)
        if previous is not None: return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No exact causal route')
    dispatch.previous = previous
    return dispatch
