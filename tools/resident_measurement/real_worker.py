"""Prepared real entrypoint. NOT device-approved. No fake/environment selector."""
import sys
from resident import worker, retained_sampler
if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('internal pipe-fd/session arguments required')
    worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], retained_sampler)
