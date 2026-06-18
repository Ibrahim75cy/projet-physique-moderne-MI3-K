####------SchrodingerNum-----###
# Partie 3.2 : Algorithme pour l'equation de Schrodinger
# Resolution numerique 1D : fonction d'onde stockee dans un tableau 2D (nx lignes, nt colonnes)
# Schema : derivee 1ere/temps (Euler explicite) + derivee 2nde/espace (differences finies centrees)
# auteur : Panayotis Akridas  fridjine ibrahim dublanc mattheiu ikram
# contributeur : Claude de Anthropic AI pour la consolidation du code
# date creation : juin 2026
# version 1
###

import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Import optionnel de PaquetOndes.py pour la confrontation (question 3.2.5)
# Si le fichier n'est pas trouve, on bascule sur la formule analytique recopiee.
try:
    from PaquetOndes import Compute_gaussian_wp
    PAQUET_DISPO = True
except Exception:
    PAQUET_DISPO = False


# ── Unites reduites (coherentes avec PaquetOndes.py) ────────────────────────
HBAR  = 1.0    # cte de Planck reduite
HBAR2 = HBAR*HBAR
MASS  = 1.0    # masse de la particule

# ── Paquet d'ondes initial (memes valeurs que PaquetOndes.py) ───────────────
K0    = 1.0    # nombre d'onde central
SIGMA = 0.2    # largeur gaussienne dans l'espace des k
X0    = -20.0  # position initiale du centre du paquet

# ── Potentiel constant V0 (question 3.2.1) ─────────────────────────────────
V0    = 0.0    # potentiel constant. V0=0 => particule libre (confrontation 3.2.5)


# ===========================================================================
# 3.2.1 — Equation de Schrodinger 1D pour un potentiel constant V0
# ===========================================================================
# i*hbar d(Psi)/dt = -(hbar^2/2m) d2(Psi)/dx2 + V0 * Psi
# Soit, en isolant la derivee temporelle :
# d(Psi)/dt = (i/hbar) [ (hbar^2/2m) d2(Psi)/dx2 - V0 * Psi ]
# On discretise : Psi est un tableau 2d Psi[i_x, j_t].


# ── Paquet d'ondes gaussien initial ─────────────────────────────────────────

def Compute_initial_wp(x_grid, sigma=SIGMA, k0=K0, x0=X0):
    """
    Paquet d'ondes gaussien normalise a t=0, centre en x0, se propageant vers
    les x croissants. sigma est ici la largeur spatiale (ecart-type en x).
    Normalisation : integrale de |psi|^2 dx = 1.
    Remarque : sert de premiere ligne du tableau 2d (question 3.2.2).
    """
    norm     = numpy.power(2.0*pi*sigma**2, -0.25)
    enveloppe = numpy.exp(-(x_grid-x0)**2 / (4.0*sigma**2))
    phase    = numpy.exp(1j*k0*x_grid)
    return norm * enveloppe * phase


# ── 3.1 (rappel) — Derivee seconde par differences finies centrees ──────────

def Compute_second_derivative(psi_line, dx):
    """
    Derivee seconde d2(psi)/dx2 par differences finies centrees a 3 points :
      psi''(x_i) ~ [psi(x_{i+1}) - 2 psi(x_i) + psi(x_{i-1})] / dx^2
    Conditions aux bords : psi nulle aux extremites (bords fixes).
    Remarque : reutilise l'algorithme de derivation de la partie 3.1.
    """
    d2psi = numpy.zeros_like(psi_line)
    d2psi[1:-1] = (psi_line[2:] - 2.0*psi_line[1:-1] + psi_line[:-2]) / dx**2
    return d2psi


# ── 3.2.4 — Un pas d'evolution temporelle (Euler explicite) ──────────────────

def Evolve_one_step(psi_line, potential, dx, dt):
    """
    Combine derivee 1ere/temps (Euler) et derivee 2nde/espace pour faire
    evoluer la fonction d'onde d'un pas de temps (question 3.2.4).

        H*psi = -(hbar^2/2m) d2(psi)/dx2 + V * psi
        psi(t+dt) = psi(t) - (i*dt/hbar) * H*psi(t)

    $potential est un tableau de meme taille que $psi_line (ici constant = V0).
    """
    d2psi = Compute_second_derivative(psi_line, dx)
    H_psi = -(HBAR2/(2.0*MASS)) * d2psi + potential * psi_line
    return psi_line - (1j*dt/HBAR) * H_psi


# ── Verification de la normalisation ────────────────────────────────────────

def Check_normalization(density_line, x_grid):
    """
    Integre la densite de probabilite |psi|^2 sur la grille (methode des trapezes).
    Doit rester proche de 1 si le schema est correct.
    """
    return numpy.trapezoid(density_line, x_grid)


# ── Position du maximum de la densite ────────────────────────────────────────

def Compute_max_position(density_line, x_grid):
    """Position du maximum de |psi|^2 (centre du paquet pour une gaussienne)."""
    return x_grid[numpy.argmax(density_line)]


# ===========================================================================
# 3.2.2 + 3.2.3 + 3.2.4 — Construction du tableau 2D et resolution
# ===========================================================================

def Solve_schrodinger_2d(x_grid, t_grid, v0=V0):
    """
    Resout l'equation de Schrodinger en remplissant un tableau 2d.

    Structure (question 3.2.2) :
      - psi2d a nx lignes (espace) et nt colonnes (temps)
      - 1ere colonne (t=0) : paquet d'ondes gaussien initial
      - reste : initialise a zero, puis rempli par l'evolution

    Methode (question 3.2.4) : on avance colonne par colonne avec Evolve_one_step.
    Renvoie psi2d de forme (nx, nt).
    """
    nx = len(x_grid)
    nt = len(t_grid)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    # tableau 2d rempli de zeros (question 3.2.2)
    psi2d = numpy.zeros((nx, nt), dtype=complex)

    # 1ere colonne = paquet d'ondes gaussien initial
    psi2d[:, 0] = Compute_initial_wp(x_grid)

    # potentiel constant V0 sur toute la grille (question 3.2.1)
    potential = v0 * numpy.ones(nx)

    # condition de stabilite indicative du schema (a discuter en 3.2.5)
    r = HBAR*dt/(2.0*MASS*dx**2)
    print(f"  parametre de stabilite r = hbar*dt/(2*m*dx^2) = {r:.4f}")

    # evolution colonne par colonne (question 3.2.4)
    print("Resolution (Euler explicite)...")
    for j in range(nt-1):
        psi2d[:, j+1] = Evolve_one_step(psi2d[:, j], potential, dx, dt)
    print("Done.")
    return psi2d


# ===========================================================================
# 3.2.5 — Confrontation au programme PaquetOndes.py (cas V0 = 0)
# ===========================================================================

def Compare_to_analytic(psi2d, x_grid, t_grid):
    """
    Confronte la resolution numerique a la solution analytique de PaquetOndes.py
    pour la particule libre (V0=0), sans representation graphique dans un premier
    temps (question 3.2.5) : on compare la norme et la position du maximum a
    quelques instants.
    """
    print("\n=== Confrontation numerique / analytique (V0=0) ===")
    print(f"{'t':>8} | {'norme num':>10} | {'x_max num':>10} | {'x_max ana':>10}")
    print("-"*48)

    indices = numpy.linspace(0, len(t_grid)-1, 6, dtype=int)
    for j in indices:
        t_val   = t_grid[j]
        density = numpy.abs(psi2d[:, j])**2
        norm    = Check_normalization(density, x_grid)
        x_num   = Compute_max_position(density, x_grid)

        # position analytique du centre : x0 + v_g * t, avec v_g = hbar k0 / m
        x_ana = X0 + (HBAR*K0/MASS)*t_val
        print(f"{t_val:8.2f} | {norm:10.5f} | {x_num:10.3f} | {x_ana:10.3f}")

    if not numpy.isfinite(numpy.abs(psi2d[:, -1]).sum()):
        print("\n/!\\ La norme diverge (nan/inf) : le schema d'Euler explicite est")
        print("    INSTABLE pour l'equation de Schrodinger. C'est le constat attendu")
        print("    de la question 3.2.5 : la condition r<1/2 vaut pour l'equation de")
        print("    la chaleur, pas pour Schrodinger (coefficient de diffusion imaginaire).")


# ── Representation graphique de la confrontation ─────────────────────────────

def Plot_comparison(psi2d, x_grid, t_grid, n_snapshots=4):
    """
    Trace |psi|^2 numerique a plusieurs instants, et la solution analytique de
    PaquetOndes.py si elle est disponible (Compute_gaussian_wp).
    """
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


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # --- 3.2.3 : tableaux 1d pour l'espace et le temps -------------------
    X_MIN, X_MAX = -40.0, 40.0
    N_X          = 2000
    x = numpy.linspace(X_MIN, X_MAX, N_X)

    dx = x[1] - x[0]
    # pas de temps respectant (indicativement) r = hbar dt/(2 m dx^2)
    r_cible = 0.4
    dt      = r_cible * 2.0*MASS*dx**2 / HBAR
    # Remarque (question 3.2.5) : quel que soit dt, le schema d'Euler explicite
    # est INSTABLE pour Schrodinger. La norme diverge rapidement (voir sortie).
    T_MAX   = 10.0
    N_T     = int(T_MAX/dt)
    t = numpy.linspace(0.0, T_MAX, N_T)

    print("=== Parametres ===")
    print(f"  nx = {N_X}, dx = {dx:.5f}")
    print(f"  nt = {N_T}, dt = {dt:.6f}")
    print(f"  V0 = {V0}  (particule libre si V0=0)")
    print(f"  PaquetOndes.py importe : {PAQUET_DISPO}")
    print("")

    # --- 3.2.2 + 3.2.4 : construction du tableau 2d et resolution --------
    psi2d = Solve_schrodinger_2d(x, t, v0=V0)

    # --- 3.2.5 : confrontation (norme + position d'abord) ----------------
    Compare_to_analytic(psi2d, x, t)

    # --- 3.2.5 : confrontation graphique ---------------------------------
    Plot_comparison(psi2d, x, t, n_snapshots=4)
