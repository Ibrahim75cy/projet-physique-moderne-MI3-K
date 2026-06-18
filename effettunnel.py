####------EffetTunnel-----###
#Partie 4 : franchissement d_une barriere de potentiel par effet tunnel
#Resolution numerique de l_equation de Schrodinger (schema de Crank-Nicolson, unitaire)
#mesure du temps de traversee tau_0 (libre) et tau_t (tunnel)
#auteur : (a completer par le groupe)
#contributeur : Claude de Anthropic AI pour la consolidation du code numerique
#date creation : 18 juin 2026
#version 1
#todo : etude systematique de l_influence de a et de V0 (boucles automatisees)
###

import numpy
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.integrate
import scipy.linalg

# ── Constantes physiques (unites reduites, coherentes avec PaquetOndes.py) ──────
HBAR  = 1.0      # cte de Planck reduite
HBAR2 = HBAR*HBAR
MASS  = 1.0      # masse de la particule
HALF  = 0.5

# ── Parametres du paquet d_ondes initial (gaussien) ─────────────────────────────
K0    = 2.0      # nombre d_onde central (energie E = HBAR2*K0**2/(2*MASS))
SIGMA = 1.5      # ecart-type *dans l_espace* du paquet initial (largeur spatiale)
X0    = -25.0    # position initiale du centre du paquet (a gauche de la barriere)

# ── Parametres de la barriere de potentiel ──────────────────────────────────────
V0       = 4.0   # hauteur de la barriere (V0>0). E = HBAR2*K0**2/2 = 2.0 ici => E<V0 : regime tunnel
A_BARR   = 1.0   # largeur (longueur) a de la barriere (assez fine pour une transmission mesurable)
X_BARR   = 0.0   # abscisse du bord gauche de la barriere

# ── Grilles espace et temps ─────────────────────────────────────────────────────
X_MIN, X_MAX = -60.0, 60.0
N_X          = 2000
x            = numpy.linspace(X_MIN, X_MAX, N_X)
DX           = x[1] - x[0]

T_MAX        = 30.0
N_T          = 1500
t_grid       = numpy.linspace(0.0, T_MAX, N_T)
DT           = t_grid[1] - t_grid[0]


# ── Energie de la particule ──────────────────────────────────────────────────────
def Compute_energy(k0=K0):
    """
    Energie cinetique moyenne associee au nombre d_onde central k0.
    E = HBAR^2 k0^2 / (2 m)
    """
    return HBAR2 * k0**2 / (2*MASS)


def Compute_kappa(v0=V0, k0=K0):
    """
    Coefficient d_attenuation kappa dans la barriere : kappa = sqrt(2 m (V0-E))/hbar.
    Remarque : n_a de sens que dans le regime tunnel E < V0 (sinon renvoie nan).
    """
    energy = Compute_energy(k0)
    return numpy.sqrt(2*MASS*(v0-energy))/HBAR if v0 > energy else float("nan")


# ── Vitesses ─────────────────────────────────────────────────────────────────────
def Compute_phase_velocity(k0=K0):
    """Calcule la vitesse de phase v_phi = hbar k0 / (2 m)."""
    return HBAR*k0/(2*MASS)

def Compute_group_velocity(k0=K0):
    """Calcule la vitesse de groupe v_g = hbar k0 / m = 2 v_phi."""
    return Compute_phase_velocity(2*k0)


# ── Potentiel ───────────────────────────────────────────────────────────────────
def Compute_potential(x_array, v0=V0, a=A_BARR, x_barr=X_BARR):
    """
    Construit le tableau du potentiel V(x) : barriere rectangulaire de hauteur v0,
    de largeur a, dont le bord gauche est en x_barr.
    Renvoie un tableau de meme taille que $x_array.
    """
    potential = numpy.zeros_like(x_array)
    inside = (x_array >= x_barr) & (x_array <= x_barr + a)
    potential[inside] = v0
    return potential


# ── Paquet d_ondes gaussien initial ──────────────────────────────────────────────
def Compute_ini_gaussian_wp(x_array, sigma=SIGMA, k0=K0, x0=X0):
    """
    Paquet d_ondes gaussien normalise a t=0, centre en x0, de largeur spatiale sigma,
    se propageant vers les x croissants (terme exp(i k0 x)).
    psi(x,0) = (2 pi sigma^2)^{-1/4} exp(-(x-x0)^2/(4 sigma^2)) exp(i k0 x)
    Remarque : la norme est verifiee numeriquement dans le main.
    """
    norm_factor = numpy.power(2*pi*sigma**2, -0.25)
    enveloppe   = numpy.exp(-(x_array-x0)**2 / (4*sigma**2))
    phase       = numpy.exp(1j*k0*x_array)
    return norm_factor * enveloppe * phase


# ── Outils de diagnostic ─────────────────────────────────────────────────────────
def Check_normalization(density_array, x_interval):
    """
    Integre la densite de probabilite sur l_intervalle (methode de Simpson).
    Doit renvoyer 1 si $density_array est bien une densite de probabilite.
    """
    return scipy.integrate.simpson(density_array, x=x_interval)


def Compute_numeric_position(density_array, x_array):
    """
    Position moyenne <x> = integrale x |psi|^2 dx (centre de masse du paquet).
    Remarque : plus robuste que le simple argmax quand le paquet se deforme.
    """
    return scipy.integrate.simpson(x_array*density_array, x=x_array)


def Compute_transmitted_probability(density_array, x_array, x_right):
    """
    Probabilite de presence a droite de la barriere : integrale de |psi|^2 pour x > x_right.
    <=> coefficient de transmission numerique du paquet.
    """
    mask = x_array > x_right
    return scipy.integrate.simpson(density_array[mask], x=x_array[mask])


# ── Coeur numerique : Crank-Nicolson ─────────────────────────────────────────────
def Build_cn_matrices(potential, dx=DX, dt=DT):
    """
    Construit les deux matrices tridiagonales du schema de Crank-Nicolson pour
    l_equation de Schrodinger i hbar d psi/dt = [-hbar^2/(2m) d2/dx2 + V] psi.

    Le schema s_ecrit : A psi^{n+1} = B psi^n  avec
        A = I + i dt/(2 hbar) H
        B = I - i dt/(2 hbar) H
    ou H est la hamiltonien discretise (laplacien a 3 points + potentiel).

    Remarque : ce schema est UNITAIRE => il conserve la norme (contrairement a Euler
    explicite qui diverge). C_est le point cle pour mesurer un temps de traversee.

    Renvoie (A, B) sous forme de matrices creuses scipy.
    """
    n = len(potential)
    # Coefficient cinetique : -hbar^2/(2 m dx^2) sur la diagonale -2, +1 lateral
    coeff_kin = HBAR2 / (2*MASS*dx**2)

    # Diagonale principale du Hamiltonien : 2*coeff_kin + V(x)
    main_diag = 2*coeff_kin + potential
    off_diag  = -coeff_kin * numpy.ones(n-1)

    # Hamiltonien tridiagonal
    hamiltonian = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], [-1, 0, 1], format="csc")

    identity = scipy.sparse.identity(n, format="csc")
    factor   = 1j*dt/(2*HBAR)

    matrix_a = identity + factor*hamiltonian
    matrix_b = identity - factor*hamiltonian
    return matrix_a, matrix_b


def Solve_schrodinger(psi_ini, potential, n_t=N_T, dx=DX, dt=DT):
    """
    Resout l_equation de Schrodinger dependante du temps par Crank-Nicolson.
    Renvoie un tableau 2d psi[n_t, n_x] : chaque ligne = la fonction d_onde a un instant.

    Methode : factorisation LU de A (constante dans le temps) puis resolution
    de A psi^{n+1} = B psi^n a chaque pas.
    """
    n_x = len(psi_ini)
    psi = numpy.empty((n_t, n_x), dtype=complex)
    psi[0] = psi_ini

    matrix_a, matrix_b = Build_cn_matrices(potential, dx, dt)
    solver = scipy.sparse.linalg.splu(matrix_a.tocsc())  # factorisation une seule fois

    print("Resolution de l_equation de Schrodinger (Crank-Nicolson)...")
    for n in range(n_t-1):
        rhs        = matrix_b.dot(psi[n])
        psi[n+1]   = solver.solve(rhs)
    print("Done.")
    return psi


# ── Mesure des temps de traversee ────────────────────────────────────────────────
def Compute_tau_free(k0=K0, a=A_BARR):
    """
    Temps analytique mis par la particule LIBRE (V0=0) pour parcourir la distance a.
    Le centre du paquet se deplace a la vitesse de groupe : tau_0,th = a / v_g.
    """
    return a / Compute_group_velocity(k0)


def Compute_tau_crossing_numeric(psi, x_array, t_array, x_barr=X_BARR, a=A_BARR):
    """
    Temps de traversee numerique de la barriere par effet tunnel.

    Definition adoptee (methode du temps de presence) :
      - instant d_entree t_in : le centre de masse <x> du paquet total atteint
        l_entree de la barriere x_barr ;
      - instant de sortie t_out : le centre de masse de la partie TRANSMISE
        (x > x_out) franchit nettement la sortie x_out, une fois qu_une
        probabilite transmise stable s_est etablie.
    tau_t,num = t_out - t_in.

    Remarque : plusieurs conventions coexistent (suivi du pic, temps de phase,
    temps de presence). Le choix est a justifier en soutenance.
    """
    x_in  = x_barr        # entree de la barriere
    x_out = x_barr + a    # sortie de la barriere

    # instant d_entree : le centre du paquet incident (parti de X0 a la vitesse
    # de groupe v_g) atteint l_entree de la barriere. Repere fiable car non pollue
    # par l_onde reflechie (contrairement au centre de masse global).
    v_g = Compute_group_velocity()
    t_entry = (x_in - X0) / v_g

    t_exit = None
    for n, t_val in enumerate(t_array):
        if t_val < t_entry:
            continue
        density = numpy.abs(psi[n])**2
        mask = x_array > x_out
        prob_trans = scipy.integrate.simpson(density[mask], x=x_array[mask])
        # on attend qu_une probabilite transmise significative se soit etablie
        if prob_trans > 1e-3:
            pos_trans = (scipy.integrate.simpson(x_array[mask]*density[mask], x=x_array[mask])
                         / prob_trans)
            if pos_trans >= x_out:
                t_exit = t_val
                break

    if t_exit is None:
        return float("nan")
    return t_exit - t_entry


# ── Etudes parametriques (clauses 4.1.d et 4.1.e) ───────────────────────────────
def Study_influence_of_a(a_values, v0=V0):
    """
    Clause 4.1.d : influence de la longueur a de la barriere sur le temps de traversee.
    Renvoie deux listes : tau_libre(a) et tau_tunnel(a).
    """
    tau_free_list   = []
    tau_tunnel_list = []
    for a in a_values:
        potential = Compute_potential(x, v0=v0, a=a)
        psi_ini   = Compute_ini_gaussian_wp(x)
        psi       = Solve_schrodinger(psi_ini, potential)
        tau_free_list.append(Compute_tau_free(a=a))
        tau_tunnel_list.append(Compute_tau_crossing_numeric(psi, x, t_grid, a=a))
    return tau_free_list, tau_tunnel_list


def Study_influence_of_v0(v0_values, a=A_BARR):
    """
    Clause 4.1.e : influence de la hauteur V0 de la barriere sur le temps de traversee tunnel.
    Renvoie la liste tau_tunnel(V0).
    """
    tau_tunnel_list = []
    for v0 in v0_values:
        potential = Compute_potential(x, v0=v0, a=a)
        psi_ini   = Compute_ini_gaussian_wp(x)
        psi       = Solve_schrodinger(psi_ini, potential)
        tau_tunnel_list.append(Compute_tau_crossing_numeric(psi, x, t_grid, a=a))
    return tau_tunnel_list


# ── Animation ────────────────────────────────────────────────────────────────────
def animate_tunnel(psi, potential, x_array, t_array, interval=20):
    """
    Anime la densite de probabilite |psi(x,t)|^2 rencontrant la barriere.
    Affiche la barriere (mise a l_echelle), la norme, et la probabilite transmise.
    """
    density_all = numpy.abs(psi)**2
    y_max = density_all[0].max() * 2.5

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Effet tunnel : paquet d_onde gaussien sur une barriere", fontsize=13)

    # Barriere (mise a l_echelle pour etre visible sur le graphe des densites)
    v_scaled = potential / (V0 if V0 != 0 else 1.0) * y_max * 0.5
    ax.fill_between(x_array, 0, v_scaled, color="#d9b38c", alpha=0.5,
                    label=f"barriere (V0={V0}, a={A_BARR})")

    (line_density,) = ax.plot(x_array, density_all[0], color="#3266ad", lw=2,
                              label=r"$|\psi(x,t)|^2$")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    time_text  = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10, color="#3266ad")
    norm_text  = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=9,  color="gray")
    trans_text = ax.text(0.02, 0.78, "", transform=ax.transAxes, fontsize=9,  color="crimson")

    x_out = X_BARR + A_BARR

    def update(frame):
        density = density_all[frame]
        line_density.set_ydata(density)
        norm  = Check_normalization(density, x_array)
        trans = Compute_transmitted_probability(density, x_array, x_out)
        time_text.set_text(f"t = {t_array[frame]:.2f}")
        norm_text.set_text(f"norme = {norm:.4f}")
        trans_text.set_text(f"prob. transmise = {trans:.4f}")
        return line_density, time_text, norm_text, trans_text

    ani = animation.FuncAnimation(fig, update, frames=len(t_array),
                                  interval=interval, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    energy = Compute_energy()
    kappa  = Compute_kappa()
    print("")
    print("=== Parametres de la simulation ===")
    print(f" - nombre d_onde central k0 = {K0}")
    print(f" - energie E                = {energy:.4f}")
    print(f" - hauteur barriere V0      = {V0}")
    print(f" - regime                   = {'TUNNEL (E<V0)' if energy < V0 else 'classique (E>V0)'}")
    print(f" - largeur barriere a       = {A_BARR}")
    print(f" - kappa (attenuation)      = {kappa:.4f}")
    print(f" - vitesse de groupe v_g    = {Compute_group_velocity():.4f}")
    print("")

    # --- Etat initial et verification de la norme -----------------------
    potential = Compute_potential(x)
    psi_ini   = Compute_ini_gaussian_wp(x)
    norm_ini  = Check_normalization(numpy.abs(psi_ini)**2, x)
    print(f"Norme initiale du paquet : {norm_ini:.6f}")

    # --- Resolution -----------------------------------------------------
    psi = Solve_schrodinger(psi_ini, potential)
    norm_fin = Check_normalization(numpy.abs(psi[-1])**2, x)
    print(f"Norme finale du paquet   : {norm_fin:.6f}  (doit rester ~1 : schema unitaire)")

    # --- Temps de traversee (clauses 4.1.b et 4.1.c) --------------------
    tau_free_th  = Compute_tau_free()
    tau_cross    = Compute_tau_crossing_numeric(psi, x, t_grid)
    prob_trans   = Compute_transmitted_probability(numpy.abs(psi[-1])**2, x, X_BARR+A_BARR)
    print("")
    print("=== Resultats ===")
    print(f" - tau_0 (libre, analytique a/v_g) = {tau_free_th:.4f}")
    print(f" - tau_t (tunnel, numerique)       = {tau_cross:.4f}")
    print(f" - probabilite transmise finale    = {prob_trans:.4f}")
    print("")

    # --- Animation ------------------------------------------------------
    animate_tunnel(psi, potential, x, t_grid, interval=15)
