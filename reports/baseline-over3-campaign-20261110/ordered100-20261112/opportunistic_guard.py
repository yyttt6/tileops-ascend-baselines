"""Fail immediately on reservation contention; retain original3s idle and all guard checks."""
import importlib.util,sys,pathlib
source=str(pathlib.Path(__file__).with_name('guard_private.py'))
spec=importlib.util.spec_from_file_location('baseline_original_guard',source)
g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
original=g.FifoTicket.try_acquire
def immediate(self,fd,metadata):
    # Do not join an existing queue, including pending ticket publication.
    if self.fd is None and any(pathlib.Path(g.LOCK_DIR/g.QUEUE_NAME).iterdir()):
        g.emit('opportunistic only: queue became nonempty; exiting')
        raise SystemExit(g.BUSY)
    if not original(self,fd,metadata):
        g.emit('opportunistic only: reservation contention; exiting without waiting')
        raise SystemExit(g.BUSY)
    return True
g.FifoTicket.try_acquire=immediate
sys.exit(g.main())
