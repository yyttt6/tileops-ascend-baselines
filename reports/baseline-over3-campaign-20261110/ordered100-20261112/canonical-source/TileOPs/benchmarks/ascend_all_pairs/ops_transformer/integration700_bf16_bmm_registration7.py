"""CPU-lazy exact seven reviewed Bmm/Grouped routes; pending parent publication."""
import builtins, hashlib, json, os, types, tempfile, shutil, sys, ast
from pathlib import Path
from types import MappingProxyType
ROOT = '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18'
MANIFEST = '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_registration7/FROZEN_PINS.json'
MANIFEST_HASH = '6208eafd82810c841f16765304af27f5d4cacb7fde404793ec2d6814b2c85bf3'
LOCAL_FILES = ['/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/resource_gate.py',
 '/root/dyq/workspace/TileOPs/benchmarks/ascend_baselines.py',
 '/root/dyq/workspace/TileOPs/benchmarks/ascend_math_more.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/binding.py',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_prep10/adapter.py',
 '/root/dyq/workspace/TileOPs/benchmarks/api.py']
CASE_PARAMS = {('ops-transformer', 'BmmFwdOp', 'mha-decode-pv-bfloat16'): '{"a":[[64,128,2048],"bfloat16"],"b":[[64,2048,128],"bfloat16"],"d":[[64,128,128],"bfloat16"]}',
 ('ops-transformer', 'BmmFwdOp', 'moe-prefill-bfloat16'): '{"a":[[128,512,2048],"bfloat16"],"b":[[128,2048,512],"bfloat16"],"d":[[128,512,512],"bfloat16"]}',
 ('ops-transformer', 'BmmFwdOp', 'square-1k-bfloat16'): '{"a":[[8,1024,1024],"bfloat16"],"b":[[8,1024,1024],"bfloat16"],"d":[[8,1024,1024],"bfloat16"]}',
 ('ops-transformer', 'BmmFwdOp', 'square-256-bfloat16'): '{"a":[[32,256,256],"bfloat16"],"b":[[32,256,256],"bfloat16"],"d":[[32,256,256],"bfloat16"]}',
 ('ops-transformer', 'BmmFwdOp', 'square-2k-bfloat16'): '{"a":[[8,2048,2048],"bfloat16"],"b":[[8,2048,2048],"bfloat16"],"d":[[8,2048,2048],"bfloat16"]}',
 ('ops-transformer', 'BmmFwdOp', 'square-512-bfloat16'): '{"a":[[16,512,512],"bfloat16"],"b":[[16,512,512],"bfloat16"],"d":[[16,512,512],"bfloat16"]}',
 ('ops-transformer', 'GroupedGemmFwdOp', 'qwen3-30b-ep8-empty-bfloat16'): '{"a":[[8,2048],"bfloat16"],"b":[[16,1536,2048],"bfloat16"],"batch_offsets":[[16],"int32"],"batch_sizes":[[16],"int32"],"output":[[8,1536],"bfloat16"],"transpose_a":false,"transpose_b":true}'}
CASE_INDICES = {('ops-transformer', 'BmmFwdOp', 'mha-decode-pv-bfloat16'): '{"B":64,"K":2048,"M":128,"N":128,"T":"bfloat16"}',
 ('ops-transformer', 'BmmFwdOp', 'moe-prefill-bfloat16'): '{"B":128,"K":2048,"M":512,"N":512,"T":"bfloat16"}',
 ('ops-transformer', 'BmmFwdOp', 'square-1k-bfloat16'): '{"B":8,"K":1024,"M":1024,"N":1024,"T":"bfloat16"}',
 ('ops-transformer', 'BmmFwdOp', 'square-256-bfloat16'): '{"B":32,"K":256,"M":256,"N":256,"T":"bfloat16"}',
 ('ops-transformer', 'BmmFwdOp', 'square-2k-bfloat16'): '{"B":8,"K":2048,"M":2048,"N":2048,"T":"bfloat16"}',
 ('ops-transformer', 'BmmFwdOp', 'square-512-bfloat16'): '{"B":16,"K":512,"M":512,"N":512,"T":"bfloat16"}',
 ('ops-transformer', 'GroupedGemmFwdOp', 'qwen3-30b-ep8-empty-bfloat16'): '{"G":16,"K":2048,"M":8,"N":1536,"T":"bfloat16","transpose_a":false,"transpose_b":true}'}
ENVELOPES = {'mha-decode-pv-bfloat16': {'cost': 2147483648,
                            'device_required': 3254779904,
                            'disk_required': 1350565888,
                            'elements_including_full_batch_weights_outputs_packing': 34603008,
                            'host_required': 3531603968},
 'moe-prefill-bfloat16': {'cost': 68719476736,
                          'device_required': 11811160064,
                          'disk_required': 3489660928,
                          'elements_including_full_batch_weights_outputs_packing': 301989888,
                          'host_required': 14227079168},
 'qwen3-30b-ep8-empty-bfloat16': {'cost': 25165824,
                                  'device_required': 3759015936,
                                  'disk_required': 1476624896,
                                  'elements_including_full_batch_weights_outputs_packing': 50360384,
                                  'host_required': 4161899008},
 'square-1k-bfloat16': {'cost': 8589934592,
                        'device_required': 2952790016,
                        'disk_required': 1275068416,
                        'elements_including_full_batch_weights_outputs_packing': 25165824,
                        'host_required': 3154116608},
 'square-256-bfloat16': {'cost': 536870912,
                         'device_required': 2348810240,
                         'disk_required': 1124073472,
                         'elements_including_full_batch_weights_outputs_packing': 6291456,
                         'host_required': 2399141888},
 'square-2k-bfloat16': {'cost': 68719476736,
                        'device_required': 5368709120,
                        'disk_required': 1879048192,
                        'elements_including_full_batch_weights_outputs_packing': 100663296,
                        'host_required': 6174015488},
 'square-512-bfloat16': {'cost': 2147483648,
                         'device_required': 2550136832,
                         'disk_required': 1174405120,
                         'elements_including_full_batch_weights_outputs_packing': 12582912,
                         'host_required': 2650800128}}
ENV = {'ASCEND_AICPU_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'ASCEND_CLASSIC_TRANSFORMER_VENDOR': '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer',
 'ASCEND_CUSTOM_OPP_PATH': '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer',
 'ASCEND_GLOBAL_LOG_LEVEL': '0',
 'ASCEND_HOME_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'ASCEND_OPP_PATH': '/usr/local/Ascend/cann-9.2.0-beta.2/opp',
 'ASCEND_TOOLKIT_HOME': '/usr/local/Ascend/cann-9.2.0-beta.2',
 'MKL_NUM_THREADS': '4',
 'OMP_NUM_THREADS': '4',
 'PYTHONDONTWRITEBYTECODE': '1'}
LOADED_VENDOR = {'/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so': 'd7be527562acb170afd0633b441f325adf9ca7bafc1c145b43b095835491e538',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so': 'dadeb0e253fc4a3172629696e0050bd6f0d2648efce0fb244625361410435fe2',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/op_tiling/liboptiling.so': 'dadeb0e253fc4a3172629696e0050bd6f0d2648efce0fb244625361410435fe2',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so': 'ee77abe18f55f9fb3c87acef08a48c44ee3751e297abf28892047b53c822ae1a',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/markupsafe/_speedups.cpython-312-x86_64-linux-gnu.so': '0ed5c27bc96bc511b307081fb18d33785eb976044c8812ef9189579f053de80d',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy.libs/libgfortran-040039e1.so.5.0.0': '14afb3129b1a8b50bc40a3b0820c7f1152ea9bc10121aab152943f7057472886',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so': '9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy.libs/libquadmath-96973f99.so.0.0.0': '934c22ded0e7d169c4d4678876c96051adf3d94545da962f60b41659b075da3b',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/core/_multiarray_tests.cpython-312-x86_64-linux-gnu.so': '0b2cbb7419ffc2f712987aabaf518a3b1d9de8406093f79dcb1d712a34ab6057',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so': '6a650810a8735cbdb988f74729973740a337645dd1585fef696e73e2dbc65beb',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/fft/_pocketfft_internal.cpython-312-x86_64-linux-gnu.so': '38d22249f351b1d50e9269c596889ffc638639ebc150c1e75fc9f55a0f331ab5',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so': '8822e772d743d405983f1b9c6b34b704dd297780890dc245454f1081adf50a4c',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-x86_64-linux-gnu.so': 'b39f7e2b4ccfd6904ae60fe1517f6bda8be7835b5bf69d54dec0d693c5e8b793',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_common.cpython-312-x86_64-linux-gnu.so': 'abd88ca8f447f22c4f51f226734d34732966bae61edd2aa25f74bfec954bee28',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-x86_64-linux-gnu.so': '5b3ef2ac8b78aa83bc869b70e30e2a70fbd3ab373c5253ed6eba9982a55fd7e2',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-x86_64-linux-gnu.so': '3619f7f91b9ee7197c2902c0e197e69b2e5dd45fb11cd22e572e9b0b886514a9',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-x86_64-linux-gnu.so': '577c1468f4fb40bb23184343e2c1b645a17d1d493641ea92c052f286dc6eb556',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-x86_64-linux-gnu.so': '2c9b1fe53ef119e3ed2acb73c802cf2996502b0fd51d4926d4047576cea57514',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-x86_64-linux-gnu.so': '5883302ce33afd54db6631adbf5e000297b8eb42c0ec894bbec318b5e6b14268',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-x86_64-linux-gnu.so': '87c5c730887943c603b16c53cf213398336ee410ef46595518093977afd5b0cb',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-x86_64-linux-gnu.so': '8d866b4b6107401ab9546044564208e0a5899c3903da09277d1c4875d98bcf0f',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/_C.cpython-312-x86_64-linux-gnu.so': '68123140eba0fc2d86966b9b2f265c690a0f32f225347f832f1828e22935196a',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libc10.so': '52d951ec184bf4ead929cfa5a24daa14cc617bf9e2180b6e551434b6f11d0b3f',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1': '78511033caddec6ccae8f4d62b94d56135f377c4cee33120e6d7df2ef499f69f',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libshm.so': 'd3539f0adf29d40515d063bac49a09dbb8619d405059e7a6d61da40cc7f5dc45',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libtorch.so': '28b3d3926e0674eda7dcdf26c27dfd05cb455d681d11cb84fdbf7e0df72f3f7b',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so': 'ae0f4bc33ffe73f4eb85b2fd03b036c68cf5ab6139995f6a2345f5962c1bbb81',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libtorch_global_deps.so': '2e5d497259a62960d95f4a7719abaeb24745429b49ffdf58e67672140205eb99',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib/libtorch_python.so': '247efcbc423fb65aa64640b96cd51672d4863413472ed2ddded6ad57a8647c67',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/_C.cpython-312-x86_64-linux-gnu.so': '8dc4a131b20e6e8f3000cc603e0c9687c68ba77ec81a416a9db042fa4df20c93',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib/libnpu_profiler.so': '833cc07018ba802ca73cbf5754389b0e19646d969aa95de8c73b58d47e07410e',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib/libtensorpipe.so': '8ecca77faf3cbc9f1a7e487edc13d5aa1cbafdd98d43848554f7710623ed6969',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so': 'f571f55ddd0a0889de1b6297625f7eb897a0bf1e554a720893c035bc8a0f514d',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/triton/_C/libtriton.so': 'bfda7c47939ff297f5c0fb465bfc6e3687834f7edd98cf58922a71b8b3f31c6b',
 '/root/dyq/workspace/.venv/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so': '957a099a4521c1f7669306fb6f79ca63fa4b9a0b4c463f7cac833a65a5c5c0cb',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so': '179dfb938b1f118558d14e04a189a400d23780c97c6ba25fdbf14dbc897bfef6',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so': '653085d077d6985cd7a6338095e59810a45e7efef0a952e1cbd7f1ca6b040547',
 '/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so': '4057480dec2fda046551d2adeed8aaa11b4cc05cbc2154f22ee555ecbda67dd9',
 '/usr/bin/python3.12': 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f',
 '/usr/lib/locale/C.utf8/LC_CTYPE': '59015993b1b671273461ef1167d7a9ad76dc090d521c91102e962217ed6b6dba',
 '/usr/lib/locale/locale-archive': '7a31de159ded19d74c73126489d3f714c1684d5b9262cddae10bbe4fd8581a2b',
 '/usr/lib/python3.12/lib-dynload/_asyncio.cpython-312-x86_64-linux-gnu.so': 'bf62de8aaab17955c9164e582ae7ea0aae3a704e2d8068ff6da12dbf29f784e5',
 '/usr/lib/python3.12/lib-dynload/_bz2.cpython-312-x86_64-linux-gnu.so': 'eff8df23cdc54d38ce9ca600d07c21ccfd7f22bd52a0720e599e6828d3e2eb76',
 '/usr/lib/python3.12/lib-dynload/_contextvars.cpython-312-x86_64-linux-gnu.so': 'da24b21e136402103121806aa5b031bc50e1576a44cd4a0e1f60bc16fd936706',
 '/usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so': 'aad3e24f186e6c0a1c7efa8f16004f951f527db86f6451d30118657384a456c1',
 '/usr/lib/python3.12/lib-dynload/_decimal.cpython-312-x86_64-linux-gnu.so': '20326fc4d39abff46814b64bf5304f8898d180ae58a66b5d60c0177ae317887d',
 '/usr/lib/python3.12/lib-dynload/_hashlib.cpython-312-x86_64-linux-gnu.so': '8dc6cda6dd9495b95a2bfb5b26e8cf06b72f36dc0213d958b9277a958287f0fe',
 '/usr/lib/python3.12/lib-dynload/_json.cpython-312-x86_64-linux-gnu.so': 'ca8e35c58f7bc8e27258c4af43651c941413bc6318903621253485b0395758a3',
 '/usr/lib/python3.12/lib-dynload/_lsprof.cpython-312-x86_64-linux-gnu.so': '16d114c88e524f93e487325e164be12dc6a24470603c808a2ca04314a3755a26',
 '/usr/lib/python3.12/lib-dynload/_lzma.cpython-312-x86_64-linux-gnu.so': 'cb298120e40f064dc047e638d911f8b3b0abf625bf1d6bd07902b46efd820fd7',
 '/usr/lib/python3.12/lib-dynload/_multiprocessing.cpython-312-x86_64-linux-gnu.so': 'd9194cd202544d7dc11b2d5c821c82d7f2b836059ebac0958f95881255c368d6',
 '/usr/lib/python3.12/lib-dynload/_posixshmem.cpython-312-x86_64-linux-gnu.so': 'bc7d2bf7f72457b50540feae0154d3a2ebf672017ff3642531023268622be2e5',
 '/usr/lib/python3.12/lib-dynload/_queue.cpython-312-x86_64-linux-gnu.so': '6e253343c58e3d83d5aca52e85dcad640d6e332d48dc4ff9bd55fcfd8be55d2b',
 '/usr/lib/python3.12/lib-dynload/_sqlite3.cpython-312-x86_64-linux-gnu.so': 'ecf54958b24c533f5c8dc67ca82cdce236aeb322864140bfe93eb8a69b729f4d',
 '/usr/lib/python3.12/lib-dynload/_ssl.cpython-312-x86_64-linux-gnu.so': '80c755c80688b92e0f32af9168a605f1283256077f35d30f6cf518252f2a6660',
 '/usr/lib/python3.12/lib-dynload/mmap.cpython-312-x86_64-linux-gnu.so': '533b42ea7b719c129a0a02fda105c74a263d11a91945457a2c03bd6cd01065bc',
 '/usr/lib/python3.12/lib-dynload/termios.cpython-312-x86_64-linux-gnu.so': '53fb6241c3b0a5177b5723a79eb0550fe2cd1867189cf9b1db0d12d272df3310',
 '/usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache': '52c227df9d53248238602c1ddaccd2c8ddc4cc6a61aa45d7c425af590b8806a5',
 '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2': 'c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da',
 '/usr/lib/x86_64-linux-gnu/libbz2.so.1.0.4': 'cc08c9f50a8009ffd6391e0116a100369b11ca9238fa52392c019e6645b122a1',
 '/usr/lib/x86_64-linux-gnu/libc.so.6': '3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf',
 '/usr/lib/x86_64-linux-gnu/libcrypto.so.3': '6866ed711dab8927e1bf82bf1cd30e828bdfc34f3cc1e83b1c9eb01ce6c37d9a',
 '/usr/lib/x86_64-linux-gnu/libdl.so.2': 'c2994527172cc23b7fb0f7d611b71e00a37da27af12c2e5e9e0dc1a742886c96',
 '/usr/lib/x86_64-linux-gnu/libexpat.so.1.9.1': '286682ecbc5e59a638963b1a4e6351e65eb32fcf4bdcb9cb7569b6a61fe06a8d',
 '/usr/lib/x86_64-linux-gnu/libffi.so.8.1.4': '00f593fe192f2851b8ce23b25cec2488d769beb5a8f63e8c9e563071e1075153',
 '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1': 'd93224d2b0dab4247598be683adca02f5cf00586f99c187579cd7e92058fb7cb',
 '/usr/lib/x86_64-linux-gnu/liblzma.so.5.4.5': '26b37d25f7ca2a2d348de8ecabacda12e921cd933b21fa069a179760db77219d',
 '/usr/lib/x86_64-linux-gnu/libm.so.6': 'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
 '/usr/lib/x86_64-linux-gnu/libpthread.so.0': '90491a21ce44efcf3bd4b65ec82501c091d7ebe7f8d9b341dd01d721001d9ecd',
 '/usr/lib/x86_64-linux-gnu/librt.so.1': '6d072187d88e092e81561a44c03a4a41397319022c6d43519757cc051c576d82',
 '/usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6': '85265a9d4afca6f4b325ceb078b669c754fb881abed4cafe91ccebe9d625d975',
 '/usr/lib/x86_64-linux-gnu/libssl.so.3': 'a40e7b67489bbd257e432c700ff8f6a1ad796425b5eccecdc2930127fdd86624',
 '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33': '1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11',
 '/usr/lib/x86_64-linux-gnu/libz.so.1.3': 'f61677c7bdb7c2aa2a243ea781bfd16afcb361cede7b4ef76127a9d8eb9a5242',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_cv.so': 'eb867b754f4377513446d5c226368184d6455d34571532c486fcfeeb74cd3d21',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_hccl.so': 'd3d8fb2115fc10baaf4b8bee7cb5f6377312159a998a91208e4696c44ede97f2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_legacy.so': 'f3a26e86b4d180012464615ebc61c6fc9f97d847782da7a4d677ba7d46a4a3f5',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_math.so': 'b3f114caf6e3d43f01dcce6b02f796091bc6af89dd91af47cba5876dfbed7e91',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_nn.so': 'eba29082d0a41361cb410a22106064cb579d3588f4a9062e6e4d58d2f2a0f43a',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_transformer.so': 'c49743ce2b1e1d48dfe3c48a91d60428690b6e46f1eed01d4fce998b8434251c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_comm_legacy.so': 'd745c869f2f0ffa5b141f7565ec09e7ee831aed08373557d2ac7e5641baaa8cf',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_cv.so': '3aebf82aceb46d2cab737e7164c2d6d93ca604d0f91190157b779483a06c55ac',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_legacy.so': 'a93d01178841e6ad63c472232dfd7f556c95d6dff099e1e6761ed213b46b3d0c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_math.so': 'a097eb69cc3985a70719b97a8c53a02e1600315e8156cd5c032415d46ae06e6f',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_nn.so': 'b0292540515ad6e5be7fc8f8ba478d298e75ec774437a38dc219694c03ded8fc',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_oam.so': '8261f09d22a173cd91b6a4ea9cd138b378a67f66c591ad080dff9f00412f4278',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_transformer.so': '2bca45f73b54fab3c42274c15a69ab00b379b818ceebb4c7c51dee984b4e7e08',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/dpu/libccl_dpu.so': 'd1fe077e7d5afe2b7ca01c2a7a20e37cd3a51f9bbf4627e09155f34c4cf59a6e',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/host_cpu/libopconstant_folding_math.so': '43a0b28a8eb92e38c1602e60d690b16cf19e4a1ba46d7ca1ebfc1d000d04116b',
 '/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_proto/lib/linux/x86_64/libopsproto.so': 'a3657f598e8861e37cfd69367365a78960d39de7dadc51b388d26641c3f3d230',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_mdl.so': '0727fd9f9c57e2201199bace87ce3fd3bd9e24010e0aec6951282fc26918e3bb',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_mdl_impl.so': 'bda882270019b0310165cf74adc7a2f842a2b9753e809b006c0ec64a02566067',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_mdl_impl_om2.so': '06571206d16a3b9c9cda9dfe7dc3642e39da340dd10b6efa5d5ff7ac7acc6c69',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_op_compiler.so': '64165f56c85699bfd2d17c006bfc0f5104921e08e1ef772454096c14df0943f4',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_op_executor.so': '2f7c124c6babccc6d94b47255d3c5fbeae35a63c43e5019cc8cc08781d1867c3',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_op_executor_impl.so': '838802629dfbf5a8f10aa4a4f25b0268fcab31f6a3a8889212dd448fba1b5b79',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_rt.so': 'e1e5169fb359f008fbbd871b1e7fc29d39e526d711d9b16c20ad7bdcd9233e7b',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_rt_impl.so': 'dee0ff9eb44d38f8adc7a5176c1c6c67be2493ee36c6a6ed0474c116ef3c609c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libacl_tdt_channel.so': 'd92b496294273c7c15a9d4c58fd83b4cf3500c3cc7e62f78f30d3576aeffe145',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaicore_utils.so': 'db423e08307a901561a4296eec173668136421d2d37605292fb9988f5d197613',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaicpu_const_folding.so': 'af7bf599ba5a2b4ec1b6d44f5f72437b6c944ebb499bdf2c7006b9422700bc68',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaicpu_engine_common.so': '8f282f961eb76a7518e09687ef26cbf908d47e191339866430a3cb05acc7d547',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_autofusion.so': '11dbaf398524f5c329303916f996d479eaf349b1796dff227373ef7068be2eb0',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_codegen.so': '1eceaae8c94d7d53de6c9445aa9ab24314492d9a53c41f2070a91031de06a91a',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_ir.so': '948aa0e77234555c8fc3e0b0030e6318fac4121a40272563962dc2c15caf4be2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_ir_register.so': '2848de7590590a0c8d986b769a57b6de962bf4d22a2d2d8cbf5d8759eebcf1b5',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_symbolizer.so': '27ff9226254f26ce00a691e22f3384da3f74b7e37378e6efe8425bad8ef9c4c5',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libaihac_symbolizer_af.so': 'bf6ebafe83d4a6c38cdc00b9409c0ee5eee88c8a5c47bb9409733f073e7f8cba',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascend_dump.so': 'c23cc357303d1fdd5540db7a9c519344fa3b922372b0bf370e60f7e48a778be2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascend_protobuf.so.3.13.0.0': '8b484235e228a587357d43dc25807aa59d8d82441e0b92347254fe6a5d43c03d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascend_trace.so': 'bf97efaf639c7051cce63e628416aa3c7e2a085fae4cf6000a626c492006ee60',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascend_watchdog.so': 'f597e9d9902150370e500cf7c308571706fa3acce17b943418929550b9752359',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascendalog.so': '76867adea47d87a7d4aadff958dd32558317a5bb04d85b3bfb55c2af828a5b4c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libascendcl.so': '4b270e626aad5d6a1e2b5444c60203185cf4684e06843573afb78e539b385fb2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libc_sec.so': '5f5248690fe3b5e686988a52c48f6ec05c9600830fde04b418efefbd4dee8393',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libconstant_folding_ops.so': 'ec106afb9e7447d804e2a09cbec44c08545134a5a40c0aa451bcdbe06882420a',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libcustom_op_runtime.so': '806d54110bcaa9feeb43473bd708a3e196bbfa61f271e1897d08a06305d8f291',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libdata_flow_base.so': '8869f052a55b1fa8f8c291e4222618351361474e24b750a0966ae9a1ea8506b4',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libdavinci_executor.so': 'a1730beb3bcbf62ca7932e82084058245e0c8e8985862c0b9a01db35b0110b82',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libdflow_runner.so': '0b936aefa3e24232006e7f6db3c124aaddb92c3fcee06e54755690fdf958aff4',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libdummy_tls.so': '7a155312f9c0f9fc385740d28f104500ea0b91d500c3a9ec3f118275a13dceac',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libeager_style_graph_builder_base.so': '58b522ffe5304f81ed4301de1a77c01c61214117eca0b38c60d85c9cc59a78e9',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/liberror_manager.so': 'f700168b3373845927e5e76ad5f4c5aa2ad7bd7cfb44d15bf8455c9c2c31c772',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libes_math.so': '2703443abdda2dca5411237bbcf72988b241346bf113549b06c32cfee82fe748',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libes_nn.so': 'ee0cb12ff3dd51b908e4fe095934a6467b698e7dca3caede8f2abf8d45f160fa',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libes_transformer.so': '2d7c0997af993ec6ddcfed7ad8f77d1353693a889cadc7c18ad32d1865613f94',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libexe_graph.so': '528b73054f7684ec168b86b8fb7b82b3ca557ed33d8183965836c1300431fbcb',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libflow_graph.so': 'c846528bad2f650a0b4540266fb3d84194129227f7ad0eeb6c102c8e54f99b13',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_common.so': 'ade4bfc55031aeb6a9c2097ba2b7a89ce7a755d270be0eb95227492afc85340d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_common_base.so': 'd5b2600c2bc96afcd78bd18b61fc6bbfe5b39d48c5958be21293bf992178ae06',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_compiler.so': 'e008db86e88a4510a86b2b242fc0d1431254f1f884cdf5ec30147c043bbb4255',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_executor.so': '49545cd38046fe6c4933db945b8977dddb1c53040c19a6afaa9f40c8c68112ac',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_runner.so': '2e550d1ac5b5fba843d51f241608cc98fcd2d6b9df0dd6736c0efcb90faaae49',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libge_runner_v2.so': '6b62eaeb4d48d04534de7e7eff856c22515bb02504ef31ddbd99ddf57469ea20',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libgert.so': '53da9c8ec11730263036dc815f4e297d016d4ff4e12a6afbdba668cf8f9f0eb8',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libgraph.so': 'a6398d5adac9462633556a71e82bc75f3cb49e6543409bdb8fc12849ddc89a35',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libgraph_af.so': 'efd668f001666201878e9504d1edb8a2dcd50672498066590dd17cd697f4079c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libgraph_base.so': '0a5408124619f7be1cb53a190da1d1a892d2081f01be94c6034af4720cbd03d7',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libgraph_base_af.so': 'acc6660217cfcebbe088399b14cce2af03bdff4816ad16097973b942a1ed485b',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl.so': '1e6415a783b4273f8cee0aeb0585bca432c78d2ba26554f753f1bcc0b3da8fa7',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl_alg.so': 'b6d82e79ed5ad39b548b892eb47a1abda60f59909eddc1f58165ee97b8656a38',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl_compat.so': '66813c6775db8005158d3bf90bbcaa4253b9ab9755fdf5be683897f29eda22e0',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl_legacy.so': '20c27d4144951c34a79b555f46f7f0558d3f41d44998441d1bc19025571adb64',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl_plf.so': '36d4f8b90a5e6cdb2d8fcc1bbce4d69287631230bceb056b784fa0459348c1f8',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhccl_v2.so': 'e98643e617dfb943aff0eecc62a0c225625cd1fd20528e08b71dcb857e69782b',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhcomm.so': 'c56c2d963ed4370132a1a8a5c9d499bde77b214c4c3c756df46b830ae31a6daa',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libhybrid_executor.so': 'd3b1e17c9de8270ec869dead269cd851da9d01fa695316958fe20a70e68b0841',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/liblowering.so': '6cf64e5af2a6d8e3bbe00a59184e31fd98fda46a316399d69d85354d1e495373',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libmetadef.so': '65824cd76b7b58f397be8bb08e8e04236785e965cf89ddd56d117347b3a6dcea',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libmmpa.so': '6293666c7d59650e4c9ee298e4915cd5fc64134c9b206a03139aa082305e92a3',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libmsprofiler.so': 'c9dc144d030f0013432cafda4e8b675354a8a7ea06cf33cd945b2d5f85977d13',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libnnopbase.so': 'f362a84c4c144f0a2cb1d8dc5266edf1dcb009f6843f0025139675c31afbd962',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libom2_executor.so': '18ed4c05f849b0e14fe753ed305d3d74b097f2bbdb9c966b49119d2d2d0cb783',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_cv.so': '43ce27a4162bf02149960b62a47aacd7d4b4a0a4c80b29ad3f32a041172d044c',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_math.so': '611d12ac698d797a5f81f2416072bb0428a898bbe1a53a5ff762feb5cfb49f29',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_nn.so': '415dffb5e72abad634f5a6d36b46e21a5e18cd18a8de3a06a5d1bec0d560c6c7',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_oam.so': 'efa4e1403a3bf100818a468ad22d52e2627cc979315c670a335061f56819093d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopapi_transformer.so': 'a444ada4809736f9fd30cb7903c29a85ebe6879bfd17dd1e60fa1837fb68d64d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libopp_registry.so': '12d25008706ff6577ca17685544491d1dc95ccc8e6119618ce635d85dfdb0e40',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libops_base.so': '39c3fc5b6a682a54f01a17e403c036369118de4afdcb89ad5a6d3c3696eb5d75',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libops_host_cpu.so': '5b7d834ca987179e25140db6740a7f9b8aa294d96eda5f630cd4cef80e65697f',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libplatform.so': '020d623b265b40332ee7bf91c630f0b771ce88655560301a5e91c0936b4d1804',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libprofapi.so': 'b3d3b9a2992df7214cffb31a34eb7772957af6f4a590d7c86debda5b2b7adc0a',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libprofimpl.so': '345977a874466fca354b8b78d37de6ef24c2c23d6cba7a239121c0f6a4d1dc8b',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libra.so': '8d4fad845a36d02b48ca6375b343c3dbe173ad8decfea1ff71494604e43eeebf',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libra_hdc.so': 'ed597ba091bf0b28dca16930e414870700cc66f0a4671a7e4495852f3b0be35e',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libra_peer.so': '3d938e82674925d88419b73bd673eba02d716986b352d5ac22b5618f47ae348d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libregister.so': '7d21a75f30c19f844100b223fdd08fada21682c268492439148b81eb6a2e6327',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/librs.so': 'd2c99e5311bfaaaf29cc167e03d4a58766e4fb1510cba810bf88e58d0820f69f',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libruntime.so': '1f64c1271a3eaae2a5429289112b229e20b7b8a0641cf8df5ab6c12bd24939c2',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libruntime_common.so': '0f5e8c2349cba94039d8280e862b557003696c687525c39808269c3e97acbb19',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libruntime_v200.so': '3fb689d4dd8a3d7c39e31e407e9a706575c8c1b6c77fa6ace80b81bb1e08d6b3',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libtls_adp.so': 'f164e523a4547d5df7232ab213883ee6d229e64b4bb4e88e4e1266659376a4fe',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libtopoaddrinfo.so': '4a3d5f61e7767a4e415907924c0b3c78800c8ff97545c707aca81311e163a64d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libtsdclient.so': '3d0c352ba2d5bddb735169d1e954f7d1ece88f1b897d61144303bd209774d9c4',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libunified_dlog.so': '83c0639956cf910146a675fa9b6d7bb9476b8cb877cf3d0d3bed4c85a5184f6d',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libxpu_tprt.so': '75fafc72560d26b2e2e551f4fd4db398f916d1acedb450d1bd891a6132851149',
 '/usr/local/Ascend/driver/lib64/driver/libascend_hal.so': '6817f8beae18403d194aa10f144dfaac4d8d208e98a64c2124b95001d9cbdc41'}
REQUIRED_LOADED = ('/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so',
 '/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64/libnnopbase.so')

ROOT = Path(ROOT)
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
    if require and not set(REQUIRED_LOADED).issubset(seen): raise RuntimeError('Missing reviewed loaded vendor libraries')


def _checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('Frozen BF16 Bmm dependency changed: ' + str(path))
    return raw

# The complete manifest is fixed by the published module, never trusted dynamically.
PINS = MappingProxyType(json.loads(_checked(MANIFEST, MANIFEST_HASH)))
CASE_PARAMS = MappingProxyType(CASE_PARAMS)
ENV = MappingProxyType(ENV)
LOADED_VENDOR = MappingProxyType(LOADED_VENDOR)

def _snapshot(runtime=False):
    _checked(MANIFEST, MANIFEST_HASH)
    data = {p: _checked(p, PINS[p]) for p in LOCAL_FILES}
    if runtime:
        for p, digest in PINS.items():
            if p in data: continue
            with open(p, 'rb') as f:
                observed = hashlib.file_digest(f, 'sha256').hexdigest()
            if observed != digest: raise RuntimeError('Frozen runtime/source changed: '+p)
    return MappingProxyType(data)

def _key(case, baseline='ops-transformer'):
    return baseline, case._call.signature.name, case.id

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def matches(case, baseline='ops-transformer'):
    try:
        k = _key(case, baseline)
        return (_canonical(case.params) == CASE_PARAMS.get(k)
                and _canonical(dict(case._call.indices)) == CASE_INDICES.get(k)
                and _canonical(dict(case._call.params)) == _canonical({n:v for n,v in json.loads(CASE_PARAMS[k]).items() if not isinstance(v,list)})
                and _canonical({n: v for n,v in case._call.tensors.items()}) ==
                    _canonical({n:v for n,v in json.loads(CASE_PARAMS[k]).items() if isinstance(v,list)}))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _load(data):
    private = {}; original = builtins.__import__
    paths = {'resource_gate': LOCAL_FILES[0], 'benchmarks.ascend_baselines': LOCAL_FILES[1],
             'benchmarks.ascend_math_more': LOCAL_FILES[2], 'binding': LOCAL_FILES[3],
             'adapter': LOCAL_FILES[4], 'benchmarks.api': LOCAL_FILES[5]}
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in paths:
            return private[name]
        return original(name, globals, locals, fromlist, level)
    for name, path in paths.items():
        mod = types.ModuleType('_bf16_bmm_verified_'+name.replace('.', '_'))
        mod.__file__ = path
        mod.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        code = data[path]
        if name == 'benchmarks.api':
            # Exact original passive dataclass only; avoid the API module's torch import.
            tree = ast.parse(code, path)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Implementation')
            code = ast.Module(body=ast.parse('import dataclasses\nfrom typing import Callable, Optional').body+[cls],type_ignores=[])
        exec(compile(code, path, 'exec'), mod.__dict__)
        private[name] = mod
    return private


def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value, (tuple,list)): return tuple(_freeze(v) for v in value)
    return value


def _frozen_case(key):
    from collections import namedtuple
    Signature = namedtuple('Signature', 'name')
    Call = namedtuple('CallSnapshot', 'signature indices tensors params')
    Case = namedtuple('CaseSnapshot', 'id params _call', rename=True)
    # namedtuple forbids _call field: provide a read-only property, with no mutation methods.
    class Snapshot(Case):
        __slots__ = ()
        _call = property(lambda self: self[2])
    p = json.loads(CASE_PARAMS[key]); ix = json.loads(CASE_INDICES[key])
    tensors = {n:v for n,v in p.items() if isinstance(v,list)}
    params = {n:v for n,v in p.items() if n not in tensors}
    return Snapshot(key[2],_freeze(p),Call(Signature(key[1]),_freeze(ix),_freeze(tensors),_freeze(params)))


def _proc(pid):
    s = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return int(s[19]), int(s[2])  # start ticks, process group


def _guard_identity():
    """Check running foreground group, live supervisor/child and held kernel lock."""
    import fcntl
    try:
        lock = Path('/run/lock/ascend-npu/device-0.lock')
        with lock.open('r') as f:
            m = json.load(f); child = m['child']; owner = os.environ.get('BF16_BMM_GUARD_OWNER')
            if not owner or m['owner'] != owner or m['phase'] != 'running' or m['device'] != 0: return None
            if m['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip(): return None
            if child['pid'] != os.getpgrp(): return None
            if _proc(child['pid']) != (child['start_ticks'], child['pid']): return None
            if _proc(m['pid'])[0] != m['start_ticks']: return None
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return None
        return (m['boot_id'],m['pid'],m['start_ticks'],m['requested_at'],child['pid'],child['start_ticks'],owner)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _require_guard():
    identity = _guard_identity()
    if identity is None: raise RuntimeError('Live foreground npu-guard with BF16_BMM_GUARD_OWNER required')
    return identity


CACHE_NAMES = ('ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR',
 'ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR')
SDK = '/usr/local/Ascend/cann-9.2.0-beta.2'
LD_PATH = str(ROOT/'install/vendors/bf16gmm_transformer/op_api/lib')+':'+':'.join((SDK+'/lib64',SDK+'/lib64/plugin/opskernel',SDK+'/lib64/plugin/nnengine',SDK+'/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64','/usr/local/Ascend/driver/lib64','/usr/local/Ascend/driver/lib64/common','/usr/local/Ascend/driver/lib64/driver'))
PYTHON_PATH = SDK+'/python/site-packages:'+SDK+'/opp/built-in/op_impl/ai_core/tbe:'

def _environment():
    for name, value in ENV.items():
        if os.environ.get(name) != value: raise RuntimeError('Original environment required: '+name)
    for name, value in (('LD_LIBRARY_PATH',LD_PATH),('PYTHONPATH',PYTHON_PATH)):
        if os.environ.get(name) != value: raise RuntimeError('Original loader environment required: '+name)
    for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None,'','0'): raise RuntimeError('Physical device 0 required')
    for name in ('LD_PRELOAD','PYTHONHOME','FLA_NPU_DISABLE_PTH','TORCH_DEVICE_BACKEND_AUTOLOAD'):
        if os.environ.get(name) not in (None, ''): raise RuntimeError('Unreviewed runtime override: '+name)
    site = Path('/root/dyq/workspace/.venv/lib/python3.12/site-packages')
    if {str(p.resolve()) for p in site.glob('*.pth')} != {p for p in PINS if p.endswith('.pth')}:
        raise RuntimeError('Python startup hooks changed')
    if any(n.rsplit('.',1)[-1] == 'gemm_cpu_bf16' for n in sys.modules):
        raise RuntimeError('Unreviewed GEMM patch module loaded')
    if sys.prefix != '/root/dyq/workspace/.venv': raise RuntimeError('Original Python environment required')
    root = Path(os.environ['BF16_BMM_RUN_ROOT']).resolve()
    if any(root == p or p in root.parents for p in (ROOT, ROOT.parent/'bf16_bmm_conv_grouped_native10', ROOT.parent/'bf16_bmm_conv_grouped_prep10')): raise RuntimeError('Original evidence cannot be a run root')
    for name in CACHE_NAMES:
        p = Path(os.environ[name]).resolve()
        if p != root and root not in p.parents: raise RuntimeError('Cache must be under explicit fresh run root: '+name)
    # Preserve all loader/backend controls across admission and reference execution.
    return (tuple(sorted(os.environ.items())),tuple(sys.path),sys.executable,sys.prefix)


def _new_run(case_id):
    root = Path(os.environ['BF16_BMM_RUN_ROOT']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='bf16-bmm-registration7-',dir=root))/case_id
    run.mkdir()
    return run


def _prepare_admission(key):
    identity = _require_guard()
    environment = _environment()
    data = _snapshot(runtime=True)
    private = _load(data)
    _loaded_vendor_pins()
    run_dir = _new_run(key[2])
    envelope = ENVELOPES[key[2]]
    host = private['resource_gate'].capacity()
    disk = shutil.disk_usage(run_dir).free
    if host['effective_available_host_bytes'] < envelope['host_required'] or disk < envelope['disk_required']:
        raise RuntimeError('Original host/disk resource envelope unavailable')
    if _require_guard() != identity: raise RuntimeError('Guard changed before device import')
    import torch, torch_npu
    free,total = torch.npu.mem_get_info()
    admission = dict(envelope,host=host,disk_free=disk,hbm_free=free,hbm_total=total,
                     stage='before_input_allocation',guard_identity=identity)
    (run_dir/'RESOURCE_ADMISSION.json').write_text(json.dumps(admission,indent=2))
    if free < envelope['device_required']: raise RuntimeError('Original HBM envelope unavailable')
    if _require_guard() != identity or _environment() != environment: raise RuntimeError('Admission identity/environment changed')
    return private,run_dir,identity,environment



CASE_INDICES = MappingProxyType(CASE_INDICES)
ENVELOPES = _freeze(ENVELOPES)

def make(case, baseline='ops-transformer'):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case,baseline): raise UnsupportedCase('Exact seven-key BF16 params, indices and tensors required')
    _snapshot()
    key = _key(case,baseline)
    frozen = _frozen_case(key)
    prepared = _prepare_admission(key) if _guard_identity() is not None else None
    def run(*args,**kwargs):
        nonlocal prepared
        selected,prepared = prepared,None
        if not matches(case,baseline) or _key(case,baseline) != key:
            raise UnsupportedCase('Case changed since construction')
        if selected is None: selected = _prepare_admission(key)
        private,run_dir,identity,environment = selected
        if _require_guard() != identity: raise RuntimeError('Admission belongs to a different guard')
        if _environment() != environment: raise RuntimeError('Environment changed since admission')
        _snapshot(runtime=True)
        _loaded_vendor_pins()
        owned=[]
        def native_factory(proxy):
            child=private['binding'].dense_fp32(proxy)
            owned.append(child)
            return child
        try:
            child = private['adapter'].candidate(frozen,native_factory=native_factory)
            _loaded_vendor_pins(require=True)
            result = child.implementation.run(*args,**kwargs)
            _loaded_vendor_pins(require=True)
            (run_dir/'loaded-maps.txt').write_text(Path('/proc/self/maps').read_text())
            return result
        finally:
            # Also owns the original native candidate if adapter construction fails.
            for native in reversed(owned):
                if native.cleanup is not None: native.cleanup()
    impl = types.SimpleNamespace(run=run,args=None,reset=None,noncomparable_reason=None)
    return Candidate(impl,'ops-transformer::aclnnGroupedMatmulV5',
        'Unchanged approved adapter and original dense_fp32; exact six BF16 Bmm and one Grouped case.',dict(PINS),
        limitations=('Original SDK/vendor, foreground guard and pre-input admission required.',),timing_mode='host_roundtrip')


def wrap(previous):
    def dispatch(case):
        if _key(case) in CASE_PARAMS: return make(case)
        if previous is not None: return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No exact BF16 Bmm/Grouped route')
    dispatch.previous = previous
    return dispatch
