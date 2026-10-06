"""
Crossover from discrete to continuous quantum geometry: running spectral dimension, lattice Dirac operators,
Connes distance.

Spectral dimension from the return probability P(tau) = (1/N) sum_k exp(-tau lambda_k):
    d_s(tau) = -2 dlnP/dln tau = 2 tau <lambda>_tau      (Gibbs average at inverse temperature tau)

Part A  2D periodic square lattice (64 x 64): exact diagonalisation of L = D - A, comparison with the analytic
        spectrum and with the infinite-lattice formula d_s = 2 d x (1 - I_1(x)/I_0(x)), x = 2 tau / a^2.
Part B  Sierpinski gasket, level 7 (3282 vertices): plateau d_s = 2 ln3/ln5 with log-periodic oscillations.
Part C  d = 4 modified dispersion E(k) = k^2 (1 + l^2 k^2) with a momentum cutoff: 0 -> 2 -> 4, and the closed form
        P ~ tau^{-1} [1 - sqrt(pi) s erfcx(s)], s = sqrt(tau)/(2 l) (no cutoff).
Part D  Lattice Dirac operators on the circle: norm-resolvent convergence of the Hodge-Dirac operator d + d*,
        failure for the naive symmetric difference (fermion doubling).
Part E  Connes distance on the square lattice for the incidence Dirac operator: a feasible upwind (eikonal)
        lower bound compared with the Euclidean and l1 distances.

Usage:  python quantum_geometry_crossover.py [--show]
Figures are written to ./figures (PNG and PDF), data to ./data (CSV); override with QGC_FIG_DIR, QGC_DATA_DIR.

Companion code for: R. Chen, "Crossover from Discrete to Continuous Quantum Geometry" (2026),
DOI: 10.5281/zenodo.23195881
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ive, erfcx, pbdv, gamma
from scipy.integrate import quad

FIG_DIR = os.environ.get("QGC_FIG_DIR", "figures")
DATA_DIR = os.environ.get("QGC_DATA_DIR", "data")
TAU = np.logspace(-4, 4, 321)


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=150)


def savedata(name, columns, cols, meta=()):
    os.makedirs(DATA_DIR, exist_ok=True)
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    with open(os.path.join(DATA_DIR, f"{name}.csv"), "w", encoding="utf-8", newline="\n") as fh:
        for line in meta:
            fh.write(f"# {line}\n")
        fh.write(",".join(columns) + "\n")
        np.savetxt(fh, arr, delimiter=",", fmt="%.10g")


def heat_kernel(lam, tau=TAU):
    """Return P(tau), d_s by log-central differences, and d_s = 2 tau <lambda>_tau."""
    lam = np.sort(lam)
    E = np.exp(-np.outer(tau, lam))
    P = E.mean(axis=1)
    mean_lam = (E * lam).mean(axis=1) / P
    lt, lP = np.log(tau), np.log(P)
    ds_fd = -2 * np.gradient(lP, lt)
    return P, ds_fd, 2 * tau * mean_lam


def laplacian(n_vertices, edges):
    A = np.zeros((n_vertices, n_vertices))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(axis=1)) - A


# ================================================================ Part A: square lattice
def part_a(L=64, d=2):
    idx = lambda x, y: (x % L) * L + (y % L)
    edges = [(idx(x, y), idx(x + 1, y)) for x in range(L) for y in range(L)] + \
            [(idx(x, y), idx(x, y + 1)) for x in range(L) for y in range(L)]
    lam = np.linalg.eigvalsh(laplacian(L * L, edges))
    th = 2 * np.pi * np.arange(L) / L
    exact = np.sort((2 - 2 * np.cos(th))[:, None] + (2 - 2 * np.cos(th))[None, :], axis=None)
    err = np.max(np.abs(np.sort(lam) - exact))
    P, ds_fd, ds_ex = heat_kernel(lam)
    x = 2 * TAU                                          # a = 1
    ds_inf = 2 * d * x * (1 - ive(1, x) / ive(0, x))
    i_max = np.argmax(ds_inf)
    print(f"[A] {L}x{L} lattice: max |numerical - analytic eigenvalue| = {err:.1e}")
    print(f"    infinite lattice: d_s max = {ds_inf[i_max]:.4f} at tau = {TAU[i_max]:.3g} (overshoots d = {d})")
    print(f"    d_s(tau=1e-4) = {ds_ex[0]:.2e} (bounded spectrum => d_s <= 2 tau lambda_max = {2e-4 * lam.max():.2e})")
    print(f"    finite-size drop: d_s(tau=1e4) = {ds_ex[-1]:.3f}; max |FD - 2 tau <lambda>| = {np.max(np.abs(ds_fd - ds_ex)[1:-1]):.1e}")
    savedata("square_lattice_heat_kernel", ["tau", "P", "ds_finite_difference", "ds_exact", "ds_infinite_lattice"],
             [TAU, P, ds_fd, ds_ex, ds_inf], [f"periodic {L}x{L} square lattice, a=1, L = D - A"])
    return lam, P, ds_ex, ds_inf


# ================================================================ Part B: Sierpinski gasket
def sierpinski(level):
    edges = set()
    def rec(x, y, s):
        if s == 1:
            a, b, c = (x, y), (x + 1, y), (x, y + 1)
            edges.update({tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))})
            return
        h = s // 2
        rec(x, y, h); rec(x + h, y, h); rec(x, y + h, h)
    rec(0, 0, 2 ** level)
    verts = sorted({v for e in edges for v in e})
    pos = {v: i for i, v in enumerate(verts)}
    return len(verts), [(pos[a], pos[b]) for a, b in edges]


def part_b(level=7):
    n, edges = sierpinski(level)
    assert n == (3 ** (level + 1) + 3) // 2
    lam = np.linalg.eigvalsh(laplacian(n, edges))
    P, ds_fd, ds_ex = heat_kernel(lam)
    ds_th, dH = 2 * np.log(3) / np.log(5), np.log(3) / np.log(2)
    lam1 = np.sort(lam)[1]
    # window of exactly three periods (factor 5^3), above the UV transient and below the finite-size scale 1/lambda_1
    means = {}
    for lo in (1.0, 2.0, 3.0):
        w = (TAU >= lo) & (TAU <= lo * 125)
        means[lo] = ds_ex[w].mean()
    w = (TAU >= 2.0) & (TAU <= 250.0)
    mean = means[2.0]
    u = np.log(TAU[w]) / np.log(5)
    grid = np.linspace(u[0], u[-1], 1024)
    yy = np.interp(grid, u, ds_ex[w])
    yy = (yy - np.polyval(np.polyfit(grid, yy, 1), grid)) * np.hanning(len(yy))
    F = np.abs(np.fft.rfft(yy)); fr = np.fft.rfftfreq(len(yy), grid[1] - grid[0])
    peak = fr[1 + np.argmax(F[1:])]
    print(f"[B] Sierpinski gasket level {level}: N = {n}, d_H = ln3/ln2 = {dH:.4f}, d_s theory = 2ln3/ln5 = {ds_th:.4f}")
    print(f"    finite-size scale 1/lambda_1 = {1 / lam1:.0f}")
    print(f"    mean d_s over three periods: tau in [1,125] {means[1.0]:.4f}, [2,250] {means[2.0]:.4f}, "
          f"[3,375] {means[3.0]:.4f}")
    print(f"    oscillation peak-to-peak on [2,250]: {np.ptp(ds_ex[w]):.3f}; dominant frequency after detrending "
          f"{peak:.2f} cycles per log_5(tau) unit (expected 1)")
    savedata("sierpinski_heat_kernel", ["tau", "P", "ds_finite_difference", "ds_exact"], [TAU, P, ds_fd, ds_ex],
             [f"Sierpinski gasket level {level}, N={n}; d_s theory 2ln3/ln5={ds_th:.6f}; d_H=ln3/ln2={dH:.6f}"])
    return lam, P, ds_ex, ds_th, (u, ds_ex[w], fr, F)


# ================================================================ Part C: modified dispersion in d = 4
TAU_C = np.logspace(-7, 6, 391)


def part_c(ell=10.0, Lam=np.pi, d=4):
    def ds_cutoff(E):
        out = []
        for t in TAU_C:
            k0 = min(Lam, 1 / np.sqrt(t))
            f0 = lambda k: k ** (d - 1) * np.exp(-t * E(k))
            f1 = lambda k: k ** (d - 1) * E(k) * np.exp(-t * E(k))
            kw = dict(limit=400, epsabs=0, epsrel=1e-11)
            tail = k0 < Lam and t * E(k0) < 700          # skip tails that underflow (exp(-700) ~ 1e-304)
            p0 = quad(f0, 0, k0, **kw)[0] + (quad(f0, k0, Lam, **kw)[0] if tail else 0)
            p1 = quad(f1, 0, k0, **kw)[0] + (quad(f1, k0, Lam, **kw)[0] if tail else 0)
            out.append(2 * t * p1 / p0)
        return np.array(out)
    ds4 = ds_cutoff(lambda k: k ** 2 * (1 + ell ** 2 * k ** 2))
    ds2 = ds_cutoff(lambda k: k ** 2)
    s = np.sqrt(TAU_C) / (2 * ell)
    lnP = -np.log(TAU_C) + np.log(1 - np.sqrt(np.pi) * s * erfcx(s))
    ds_closed = -2 * np.gradient(lnP, np.log(TAU_C))
    # check of the parabolic-cylinder integral (Gradshteyn-Ryzhik 3.462.1) for nu = 3/2
    nu, beta, gam = 1.5, 0.7, 1.3
    lhs = quad(lambda u: u ** (nu - 1) * np.exp(-beta * u ** 2 - gam * u), 0, np.inf)[0]
    rhs = (2 * beta) ** (-nu / 2) * gamma(nu) * np.exp(gam ** 2 / (8 * beta)) * pbdv(-nu, gam / np.sqrt(2 * beta))[0]
    at = lambda t: np.argmin(np.abs(TAU_C - t))
    mid = (TAU_C > 1e-2) & (TAU_C < 1)
    print(f"[C] d=4, E = k^2(1+l^2 k^2), l = {ell}, cutoff pi: d_s(1e-7) = {ds4[0]:.4f}, "
          f"plateau (1e-2..1) = {ds4[mid].mean():.3f}, d_s(1e4) = {ds4[at(1e4)]:.3f}, d_s(1e6) = {ds4[-1]:.4f}")
    print(f"    closed form (no cutoff) vs cutoff numerics for tau >= 1e-2: max diff = "
          f"{np.max(np.abs(ds_closed - ds4)[TAU_C >= 1e-2]):.1e}")
    print(f"    E = k^2 with cutoff: d_s(1e-7) = {ds2[0]:.4f}, max = {ds2.max():.3f}, d_s(1e6) = {ds2[-1]:.4f}")
    print(f"    parabolic-cylinder identity check: integral = {lhs:.12f}, formula = {rhs:.12f}")
    savedata("modified_dispersion_d4", ["tau", "ds_alpha4_cutoff", "ds_alpha2_cutoff", "ds_alpha4_closed_form"],
             [TAU_C, ds4, ds2, ds_closed], [f"d=4, E=k^2(1+l^2k^2), l={ell}, ball cutoff |k|<=pi (a=1)"])
    return ds4, ds2, ds_closed


# ================================================================ Part D: lattice Dirac operators
def part_d(z=1j):
    rows = []
    for N in [16, 64, 256, 1024, 4096]:
        a = 2 * np.pi / N
        k = np.arange(-N // 2 + 1, N // 2 + 1)
        tail = 1 / abs(np.sqrt((N / 2) ** 2) - abs(z))         # continuum modes outside the zone
        errs = []
        for s in [(2 / a) * np.abs(np.sin(k * a / 2)), np.abs(np.sin(k * a)) / a]:
            d = max(np.max(np.abs(1 / (sg * s - z) - 1 / (sg * np.abs(k) - z))) for sg in (1, -1))
            errs.append(max(d, tail))
        rows.append([N, a, errs[0], errs[1]])
        print(f"[D] N={N:5d}: sup resolvent error  Hodge-Dirac = {errs[0]:.2e}   naive = {errs[1]:.2e}")
    R = np.array(rows)
    savedata("dirac_resolvent_convergence", ["N", "a", "err_hodge_dirac", "err_naive"], [R[:, i] for i in range(4)],
             ["circle of length 2 pi, z = i; sup over Fourier modes of |(D_a - z)^-1 - (D - z)^-1|"])
    return R


# ================================================================ Part E: Connes distance lower bound
def part_e():
    """Incidence Dirac operator with edges oriented along +x, +y: ||[D,f]||^2 = max_v sum_{e->v} |grad_e f|^2.
    Upwind recursion f(v) = (A+B)/2 + sqrt(2-(A-B)^2)/2 gives a feasible f, hence a lower bound."""
    rows = []
    for n in [50, 200, 800]:
        f = np.zeros((n + 1, n + 1))
        f[1:, 0] = np.arange(1, n + 1)
        f[0, 1:] = np.arange(1, n + 1)
        worst = 0.0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                a_, b_ = f[i - 1, j], f[i, j - 1]
                f[i, j] = (a_ + b_) / 2 + np.sqrt(max(2 - (a_ - b_) ** 2, 0.0)) / 2
                worst = max(worst, (f[i, j] - a_) ** 2 + (f[i, j] - b_) ** 2)
        assert worst <= 1 + 1e-9, "upwind f is not feasible"
        for deg in [0, 15, 30, 45]:
            th = np.radians(deg)
            r = n / max(np.cos(th), np.sin(th))          # lattice point on the ray, at the edge of the grid
            i, j = int(round(r * np.cos(th))), int(round(r * np.sin(th)))
            eu, l1 = np.hypot(i, j), i + j
            rows.append([n, deg, f[i, j] / eu, l1 / eu])
    R = np.array(rows)
    for n in [50, 200, 800]:
        sel = R[:, 0] == n
        print(f"[E] n={n:4d}: lower bound / Euclidean at 0,15,30,45 deg = "
              + ", ".join(f"{x:.4f}" for x in R[sel, 2]) + "   (l1/Euclidean = "
              + ", ".join(f"{x:.3f}" for x in R[sel, 3]) + ")")
    savedata("connes_distance_bounds", ["n", "angle_deg", "lower_bound_over_euclid", "l1_over_euclid"],
             [R[:, i] for i in range(4)], ["square lattice, incidence Dirac operator; upwind feasible f"])
    return R


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    lamA, PA, dsA, dsInf = part_a()
    lamB, PB, dsB, dsth, (uB, dsBw, frB, FB) = part_b()
    ds4, ds2, dsC = part_c()
    RD = part_d()
    RE = part_e()

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for lam, lab, c in [(lamA, "square lattice 64x64", "tab:blue"), (lamB, "Sierpinski gasket (level 7)", "tab:red")]:
        lp = np.sort(lam[lam > 1e-9])
        ax[0, 0].loglog(lp, np.arange(1, len(lp) + 1) / len(lam), color=c, label=lab)
    ax[0, 0].set_xlabel("lambda"); ax[0, 0].set_ylabel("integrated density N(lambda)/N")
    ax[0, 0].set_title("Graph Laplacian spectra (N(lambda) ~ lambda^{d_s/2})"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].loglog(TAU, PA, color="tab:blue", label="square lattice")
    ax[0, 1].loglog(TAU, PB, color="tab:red", label="Sierpinski gasket")
    ax[0, 1].set_xlabel("tau"); ax[0, 1].set_ylabel("P(tau)"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("Return probability P(tau) = (1/N) sum exp(-tau lambda)")
    a = ax[1, 0]
    a.semilogx(TAU, dsA, color="tab:blue", label="64x64 lattice (exact diagonalisation)")
    a.semilogx(TAU, dsInf, "k:", label="infinite lattice: 2d x(1 - I1/I0)")
    a.semilogx(TAU, dsB, color="tab:red", label="Sierpinski gasket")
    a.axhline(dsth, color="tab:red", ls="--", lw=0.8, label="2 ln3/ln5")
    a.axhline(2, color="grey", lw=0.5)
    a.set_xlabel("tau"); a.set_ylabel("d_s(tau)"); a.set_ylim(0, 2.6); a.legend(fontsize=7)
    a.set_title("Running spectral dimension: UV zero, overshoot, plateau, finite-size drop")
    a = ax[1, 1]
    a.semilogx(TAU_C, ds4, color="tab:green", label="E = k^2(1+l^2k^2), cutoff pi")
    a.semilogx(TAU_C, dsC, "k:", label="closed form, no cutoff")
    a.semilogx(TAU_C, ds2, color="tab:gray", label="E = k^2, cutoff pi")
    for yv in (2, 4):
        a.axhline(yv, color="grey", lw=0.5)
    a.set_xlabel("tau"); a.set_ylabel("d_s(tau)"); a.set_ylim(0, 4.6); a.legend(fontsize=8)
    a.set_title("d = 4: discrete UV (0) -> 2 -> 4, l = 10 a")
    savefig(fig, "quantum_geometry_crossover")

    fig2, bx = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    bx[0].plot(uB, dsBw, color="tab:red"); bx[0].axhline(dsth, color="k", ls="--", lw=0.8)
    bx[0].set_xlabel("log_5 tau"); bx[0].set_ylabel("d_s"); bx[0].set_title("Sierpinski gasket: log-periodic d_s (period ln 5)")
    bx[1].loglog(RD[:, 0], RD[:, 2], "o-", label="Hodge-Dirac d + d*")
    bx[1].loglog(RD[:, 0], RD[:, 3], "s-", label="naive symmetric difference")
    bx[1].set_xlabel("N = 2 pi / a"); bx[1].set_ylabel("sup_k |resolvent difference| at z = i")
    bx[1].set_title("Norm-resolvent convergence vs fermion doubling"); bx[1].legend(fontsize=8)
    for n, mk in [(50, "o"), (200, "s"), (800, "^")]:
        sel = RE[:, 0] == n
        bx[2].plot(RE[sel, 1], RE[sel, 2], mk + "-", label=f"lower bound, n={n}")
    bx[2].plot(RE[RE[:, 0] == 800, 1], RE[RE[:, 0] == 800, 3], "k--", label="l1 / Euclidean")
    bx[2].axhline(1, color="grey", lw=0.6)
    bx[2].set_xlabel("angle (deg)"); bx[2].set_ylabel("distance / Euclidean")
    bx[2].set_title("Connes distance on the square lattice"); bx[2].legend(fontsize=8)
    savefig(fig2, "dirac_and_connes")
    print(f"figures written to {FIG_DIR}/, data to {DATA_DIR}/")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
