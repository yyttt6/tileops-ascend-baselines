"""Remove only redundant task arrays from summaries; full task evidence stays in raw."""
DROP=frozenset(('profile_kernel_tasks','profile_excluded_kernel_tasks'))
def compact(value):
 if isinstance(value,dict):
  for key in list(value):
   if key in DROP:value.pop(key)
   else:compact(value[key])
 elif isinstance(value,list):
  for item in value:compact(item)
 return value
