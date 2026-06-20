# Partie 3.2 : Algorithme pour l'equation de Schrodinger

import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Import optionnel de PaquetOndes.py pour la confrontation analytique (question 3.2.5)
try:
    from paquetondegauss1d3k import Compute_gaussian_wp
    PAQUET_DISPO = True
except Exception:
    PAQUET_DISPO = False


# Constantes physiques en unites reduites (hbar = m = 1)
HBAR  = 1.0    
HBAR2 = HBAR*HBAR
MASS  = 1.0    

# Parametres du paquet d'ondes initial
K0    = 1.0   
SIGMA = 0.2   
X0    = -20.0 

# Potentiel constant : V0 = 0,  particule libre (3.2.5)
V0    = 0.0    


# Paquet d'ondes gaussien normalise a t=0, centre en x0
def Compute_initial_wp(x_grid, sigma=SIGMA, k0=K0, x0=X0):
    norm     = numpy.power(2.0*pi*sigma**2, -0.25)
    enveloppe = numpy.exp(-(x_grid-x0)**2 / (4.0*sigma**2))
    phase    = numpy.exp(1j*k0*x_grid)
    return norm * enveloppe * phase


# Derivee seconde par differences finies centrees a 3 points (bords fixes a zero)
def Compute_second_derivative(psi_line, dx):
    d2psi = numpy.zeros_like(psi_line)
    d2psi[1:-1] = (psi_line[2:] - 2.0*psi_line[1:-1] + psi_line[:-2]) / dx**2
    return d2psi


# Un pas de temps par la methode d'Euler : Psi(t+dt) = Psi(t) - (i*dt/hbar) H*Psi
def Evolve_one_step(psi_line, potential, dx, dt):
    d2psi = Compute_second_derivative(psi_line, dx)
    H_psi = -(HBAR2/(2.0*MASS)) * d2psi + potential * psi_line
    return psi_line - (1j*dt/HBAR) * H_psi


# Integre |Psi|^2 (methode des trapezes) : doit rester proche de 1
def Check_normalization(density_line, x_grid):
    return numpy.trapezoid(density_line, x_grid)

# Position du maximum de |Psi|^2 (centre du paquet)
def Compute_max_position(density_line, x_grid):
    return x_grid[numpy.argmax(density_line)]



# Remplit le tableau 2D psi2d[espace, temps] et fait evoluer le paquet (Euler)
def Solve_schrodinger_2d(x_grid, t_grid, v0=V0):
    nx = len(x_grid)
    nt = len(t_grid)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    psi2d = numpy.zeros((nx, nt), dtype=complex)
    psi2d[:, 0] = Compute_initial_wp(x_grid)
    
    potential = v0 * numpy.ones(nx)

    r = HBAR*dt/(2.0*MASS*dx**2)
    print(f"  parametre de stabilite r = hbar*dt/(2*m*dx^2) = {r:.4f}")

    print("Resolution (Euler explicite)...")
    for j in range(nt-1):
        psi2d[:, j+1] = Evolve_one_step(psi2d[:, j], potential, dx, dt)
    print("Done.")
    return psi2d


# Confronte le numerique a la theorie (V0=0) : norme et position du maximum
def Compare_to_analytic(psi2d, x_grid, t_grid):
    print("\n=== Confrontation numerique / analytique (V0=0) ===")
    print(f"{'t':>8} | {'norme num':>10} | {'x_max num':>10} | {'x_max ana':>10}")
    print("-"*48)

    indices = numpy.linspace(0, len(t_grid)-1, 6, dtype=int)
    for j in indices:
        t_val   = t_grid[j]
        density = numpy.abs(psi2d[:, j])**2
        norm    = Check_normalization(density, x_grid)
        x_num   = Compute_max_position(density, x_grid)

        x_ana = X0 + (HBAR*K0/MASS)*t_val
        print(f"{t_val:8.2f} | {norm:10.5f} | {x_num:10.3f} | {x_ana:10.3f}")

    if not numpy.isfinite(numpy.abs(psi2d[:, -1]).sum()):
        print("\n/!\\ La norme diverge (nan/inf) : le schema d'Euler explicite est")
        print("    INSTABLE pour l'equation de Schrodinger. C'est le constat attendu")
        print("    de la question 3.2.5 : la condition r<1/2 vaut pour l'equation de")
        print("    la chaleur, pas pour Schrodinger (coefficient de diffusion imaginaire).")


# Trace |Psi|^2 a plusieurs instants (+ solution analytique si disponible)
def Plot_comparison(psi2d, x_grid, t_grid, n_snapshots=4):
    indices = numpy.linspace(0, len(t_grid)-1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(n_snapshots, 1, figsize=(10, 2.5*n_snapshots), sharex=True)

    for ax, j in zip(axes, indices):
        t_val   = t_grid[j]
        density = numpy.abs(psi2d[:, j])**2
        ax.plot(x_grid, density, color="#3266ad", lw=2, label="numerique")

        if PAQUET_DISPO:
            psi_ana = Compute_gaussian_wp(x_grid, t_val)
            ax.plot(x_grid, numpy.abs(psi_ana)**2, color="crimson", lw=1.2,
                    ls="--", label="analytique (PaquetOndes.py)")

        ax.set_ylabel(r"$|\psi|^2$")
        ax.set_title(f"t = {t_val:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    fig.suptitle("3.2.5 — Confrontation numerique / analytique (V0=0)", fontsize=13)
    plt.tight_layout()
    plt.show()

# Main

if __name__ == "__main__":

    # Grille d'espace (3.2.3)
    X_MIN, X_MAX = -40.0, 40.0
    N_X          = 2000
    x = numpy.linspace(X_MIN, X_MAX, N_X)

    dx = x[1] - x[0]
    r_cible = 0.4
    dt      = r_cible * 2.0*MASS*dx**2 / HBAR
    T_MAX   = 10.0
    N_T     = int(T_MAX/dt)
    t = numpy.linspace(0.0, T_MAX, N_T)

    print("=== Parametres ===")
    print(f"  nx = {N_X}, dx = {dx:.5f}")
    print(f"  nt = {N_T}, dt = {dt:.6f}")
    print(f"  V0 = {V0}  (particule libre si V0=0)")
    print(f"  PaquetOndes.py importe : {PAQUET_DISPO}")
    print("")

    psi2d = Solve_schrodinger_2d(x, t, v0=V0)
    Compare_to_analytic(psi2d, x, t)
    Plot_comparison(psi2d, x, t, n_snapshots=4)
