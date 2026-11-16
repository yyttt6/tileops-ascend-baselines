from pathlib import Path
from selective_scan_denominator_frozen_policy import bind
def build(case):
 reference,checker,proof=bind(case)
 proof['frozen_FP64_oracle_checker_retained']=True
 return None,reference,checker,proof
