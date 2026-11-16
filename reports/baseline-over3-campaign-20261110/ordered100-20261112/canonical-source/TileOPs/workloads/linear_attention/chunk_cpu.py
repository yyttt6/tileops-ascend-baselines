"""Independent CPU mathematics for the manifest's dense chunk training calls.

No FLA, TileOPs kernel, TileLang, Triton or device-runtime imports. All arithmetic
is FP64; public tensors are cast once to the manifest storage dtypes. This models
the real-valued operator, not a device kernel's intermediate rounding schedule.
See the accompanying CONTRACT.md for equations, evidence and acceptance limits.
"""
import torch


def _cpu(*tensors):
    for tensor in tensors:
        if tensor is not None and (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"):
            raise ValueError("CPU oracle requires CPU tensors")


def _delta_shape(q, k, v, beta, chunk_size):
    _cpu(q, k, v, beta)
    if q.ndim != 4 or q.shape != k.shape or v.shape[:-1] != q.shape[:-1] or beta.shape != q.shape[:-1]:
        raise ValueError("DeltaNet expects BHLD q/k/v and BHL beta")
    if chunk_size <= 0 or q.shape[2] == 0 or q.shape[2] % chunk_size:
        raise ValueError("The manifest requires a positive sequence divisible by chunk_size")


def _delta_chunk(k, v, beta):
    """Solve the unit lower triangular system, independently of Neumann kernels."""
    C = k.shape[-2]
    eye = torch.eye(C, dtype=k.dtype, device="cpu").expand(*k.shape[:-2], C, C)
    lower = eye + torch.tril(beta[..., None] * (k @ k.transpose(-1, -2)), diagonal=-1)
    inverse = torch.linalg.solve_triangular(lower, eye, upper=False, unitriangular=True)
    # Solve the two RHSs directly, rather than using the reported inverse as oracle.
    w = torch.linalg.solve_triangular(lower, beta[..., None] * k, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(lower, beta[..., None] * v, upper=False, unitriangular=True)
    return inverse, w, u


def delta_chunk_math(q, k, v, beta, chunk_size):
    """Return all six mathematical outputs in FP64, BHLD / BH(NC+1)KV."""
    _delta_shape(q, k, v, beta, chunk_size)
    q, k, v, beta = (x.detach().double() for x in (q, k, v, beta))
    B, H, L, K = q.shape
    V = v.shape[-1]
    state = torch.zeros((B, H, K, V), dtype=torch.float64, device="cpu")
    states, outputs, inverses, ws, us = [state], [], [], [], []
    for start in range(0, L, chunk_size):
        sl = slice(start, start + chunk_size)
        qc, kc, vc, bc = q[:, :, sl], k[:, :, sl], v[:, :, sl], beta[:, :, sl]
        inverse, w, u = _delta_chunk(kc, vc, bc)
        residual = u - w @ state
        outputs.append(qc @ state + torch.tril(qc @ kc.transpose(-1, -2)) @ residual)
        state = state + kc.transpose(-1, -2) @ residual
        states.append(state)
        inverses.append(inverse)
        ws.append(w)
        us.append(u)
    A = torch.cat(inverses, dim=2)
    return (torch.cat(outputs, dim=2), torch.stack(states, dim=2), A, A.clone(),
            torch.cat(ws, dim=2), torch.cat(us, dim=2))


@torch.no_grad()
def deltanet_chunk_fwd_cpu(q, k, v, beta, chunk_size):
    """(o, S, Aw, Au, w, u); S is FP32, the other five use q.dtype.

    q is deliberately NOT scaled by K**-0.5: that is the chunk-training contract.
    S contains the zero initial state, each boundary, and the final state.
    """
    result = delta_chunk_math(q, k, v, beta, chunk_size)
    return tuple(x.to(torch.float32 if i == 1 else q.dtype).contiguous() for i, x in enumerate(result))


@torch.no_grad()
def deltanet_saved_cpu(q, k, v, beta, chunk_size):
    """Generate the five backward inputs in their declared order and dtype."""
    # All six results have a defined oracle; this convenience method selects only
    # the five tensors declared as backward inputs. It is not a forward verifier.
    return deltanet_chunk_fwd_cpu(q, k, v, beta, chunk_size)[1:]


def _gla_shape(k, v, g, chunk_size, initial_state):
    _cpu(k, v, g, initial_state)
    if k.ndim != 4 or k.shape != g.shape or v.shape[:-1] != k.shape[:-1]:
        raise ValueError("GLA expects BTHK k/g and BTHV v")
    B, T, H, K = k.shape
    if chunk_size <= 0 or T == 0 or T % chunk_size:
        raise ValueError("The manifest requires a positive sequence divisible by chunk_size")
    if initial_state is not None and initial_state.shape != (B, H, K, v.shape[-1]):
        raise ValueError("GLA initial_state must be BHKV")


def gla_states_math(k, v, g, chunk_size, initial_state=None):
    """Exact chunk boundaries, B(NC+1)HKV, with no exp(-prefix) overflow."""
    _gla_shape(k, v, g, chunk_size, initial_state)
    k, v, g = (x.detach().double() for x in (k, v, g))
    B, T, H, K = k.shape
    state = (torch.zeros((B, H, K, v.shape[-1]), dtype=torch.float64, device="cpu")
             if initial_state is None else initial_state.detach().double().clone())
    states = [state]
    for start in range(0, T, chunk_size):
        sl = slice(start, start + chunk_size)
        prefix = g[:, sl].cumsum(dim=1)
        last = prefix[:, -1]
        weighted_k = k[:, sl] * torch.exp(last[:, None] - prefix)
        state = state * last.exp()[..., None] + torch.einsum("bthk,bthv->bhkv", weighted_k, v[:, sl])
        states.append(state)
    return torch.stack(states, dim=1)


@torch.no_grad()
def gla_chunk_states_cpu(k, v, g, chunk_size, initial_state=None):
    """FP32 state entering each chunk followed by the final state."""
    return gla_states_math(k, v, g, chunk_size, initial_state).float().contiguous()


@torch.no_grad()
def delta_backward_math(do, q, k, v, beta, chunk_size):
    """Explicit reverse recurrence; all four derivatives are FP64.

    Redundant saved buffers are regenerated from original inputs. Keeping only
    chunk boundaries and one chunk's token states bounds temporary state memory.
    """
    _delta_shape(q, k, v, beta, chunk_size)
    _cpu(do)
    if do.shape != v.shape:
        raise ValueError("do must have v's shape")
    q, k, v, beta, do = (x.detach().double() for x in (q, k, v, beta, do))
    B, H, L, K = q.shape
    state = torch.zeros((B, H, K, v.shape[-1]), dtype=torch.float64, device="cpu")
    boundaries = [state]
    for start in range(0, L, chunk_size):
        sl = slice(start, start + chunk_size)
        kc = k[:, :, sl]
        _, w, u = _delta_chunk(kc, v[:, :, sl], beta[:, :, sl])
        state = state + kc.transpose(-1, -2) @ (u - w @ state)
        boundaries.append(state)
    dq, dk, dv, dbeta = (torch.empty_like(x) for x in (q, k, v, beta))
    adjoint = torch.zeros_like(state)
    for c in reversed(range(L // chunk_size)):
        states, errors = [boundaries[c]], []
        for t in range(c * chunk_size, (c + 1) * chunk_size):
            kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t]
            error = vt - torch.einsum("bhk,bhkv->bhv", kt, states[-1])
            states.append(states[-1] + kt[..., None] * (bt[..., None] * error)[..., None, :])
            errors.append(error)
        for j in reversed(range(chunk_size)):
            t = c * chunk_size + j
            qt, kt, bt, dot = q[:, :, t], k[:, :, t], beta[:, :, t], do[:, :, t]
            previous, current, error = states[j], states[j + 1], errors[j]
            dq[:, :, t] = torch.einsum("bhkv,bhv->bhk", current, dot)
            adjoint = adjoint + qt[..., None] * dot[..., None, :]
            dr = torch.einsum("bhkv,bhk->bhv", adjoint, kt)
            dvalue = bt[..., None] * dr
            dk[:, :, t] = (torch.einsum("bhkv,bhv->bhk", adjoint, bt[..., None] * error)
                           - torch.einsum("bhkv,bhv->bhk", previous, dvalue))
            dv[:, :, t] = dvalue
            dbeta[:, :, t] = (dr * error).sum(dim=-1)
            adjoint = adjoint - kt[..., None] * dvalue[..., None, :]
    return dq, dk, dv, dbeta


@torch.no_grad()
def deltanet_chunk_bwd_cpu(do, q, k, v, beta, chunk_size):
    return tuple(x.to(q.dtype).contiguous() for x in delta_backward_math(do, q, k, v, beta, chunk_size))


@torch.no_grad()
def gla_backward_math(q, k, v, g, do, chunk_size, scale=-1.0, *, initial_state=None, dht=None):
    """Explicit GLA reverse recurrence, including a final-state loss, in FP64."""
    _gla_shape(k, v, g, chunk_size, initial_state)
    _cpu(q, do, dht)
    if q.shape != k.shape or do.shape != v.shape:
        raise ValueError("q/k and do/v shapes must agree")
    q, k, v, g, do = (x.detach().double() for x in (q, k, v, g, do))
    B, T, H, K = q.shape
    sc = K ** -0.5 if scale is None or scale <= 0 else scale
    boundaries = gla_states_math(k, v, g, chunk_size, initial_state)
    adjoint = torch.zeros_like(boundaries[:, 0]) if dht is None else dht.detach().double().clone()
    if adjoint.shape != boundaries[:, 0].shape:
        raise ValueError("dht must have BHKV shape")
    dq, dk, dv, dg = (torch.empty_like(x) for x in (q, k, v, g))
    for c in reversed(range(T // chunk_size)):
        states = [boundaries[:, c]]
        for t in range(c * chunk_size, (c + 1) * chunk_size):
            states.append(states[-1] * g[:, t].exp()[..., None] + k[:, t, ..., None] * v[:, t, :, None, :])
        for j in reversed(range(chunk_size)):
            t = c * chunk_size + j
            qt, kt, vt, dot = q[:, t], k[:, t], v[:, t], do[:, t]
            dq[:, t] = sc * torch.einsum("bhkv,bhv->bhk", states[j + 1], dot)
            adjoint = adjoint + sc * qt[..., None] * dot[..., None, :]
            dk[:, t] = torch.einsum("bhkv,bhv->bhk", adjoint, vt)
            dv[:, t] = torch.einsum("bhkv,bhk->bhv", adjoint, kt)
            dg[:, t] = (adjoint * states[j]).sum(dim=-1) * g[:, t].exp()
            adjoint = adjoint * g[:, t].exp()[..., None]
    return dq, dk, dv, dg


@torch.no_grad()
def gla_chunk_bwd_cpu(q, k, v, g, do, chunk_size, scale=-1.0, *, initial_state=None, dht=None):
    return tuple(x.float().contiguous() for x in gla_backward_math(
        q, k, v, g, do, chunk_size, scale, initial_state=initial_state, dht=dht))


def validate_delta_outputs(got, expected, *, atol, rtol, normalized_bound=1e-3):
    """Check all six, including the inverse's small strict-lower entries.

    Keeps the existing forward elementwise bounds, adds the repository's usual
    normalized squared error bound. Device calibration remains a local follow-up.
    """
    if not isinstance(got, (tuple, list)) or not isinstance(expected, (tuple, list)) or len(got) != 6 or len(expected) != 6:
        raise AssertionError("DeltaNetChunkFwdOp must return all six declared tensors")
    _cpu(*got, *expected)

    def compare(a, b, *, unit_dims):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError("Output shape/dtype differs from the manifest")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("Nonfinite values are not accepted for these finite cases")
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        a, b = a.double(), b.double()
        error, energy = ((a - b) ** 2).sum(), (a ** 2 + b ** 2).sum()
        if error > normalized_bound * energy:
            raise AssertionError("Normalized squared error exceeds bound")
        # Do not let unrelated heads/chunks/tokens hide a missing local result.
        # S is checked per boundary matrix; vectors and inverse rows per row.
        unit_error = ((a - b) ** 2).sum(dim=unit_dims)
        unit_energy = (a ** 2 + b ** 2).sum(dim=unit_dims)
        if torch.any(unit_error > normalized_bound * unit_energy):
            raise AssertionError("Local normalized squared error exceeds bound")

    for i, (a, b) in enumerate(zip(got, expected, strict=True)):
        compare(a, b, unit_dims=(-2, -1) if i == 1 else (-1,))
        if i in (2, 3):
            C = b.shape[-1]
            aa, bb = a.reshape(-1, C, C), b.reshape(-1, C, C)
            compare(torch.tril(aa, diagonal=-1), torch.tril(bb, diagonal=-1), unit_dims=(-1,))
            # Unit diagonal and zero upper triangle are exact algebraic invariants.
            if not torch.equal(aa.diagonal(dim1=-2, dim2=-1), torch.ones_like(aa.diagonal(dim1=-2, dim2=-1))):
                raise AssertionError("WY inverse diagonal must be one")
            if torch.count_nonzero(torch.triu(aa, diagonal=1)):
                raise AssertionError("WY inverse upper triangle must be zero")
        if i == 1 and torch.count_nonzero(a[:, :, 0]):
            raise AssertionError("DeltaNet chunk training must start from zero state")
