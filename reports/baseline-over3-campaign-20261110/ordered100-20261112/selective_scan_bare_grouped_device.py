from selective_scan_bare_device import build as original_build
from grouped_scan_layout import build as layout_build
def build(case):
 original,reference,checker,proof=original_build(case)
 measured,layout=layout_build(case)
 proof.update(layout)
 proof['frozen_FP64_oracle_checker_retained']=True
 return measured,reference,checker,proof
