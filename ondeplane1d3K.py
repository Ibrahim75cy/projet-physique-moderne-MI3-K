####------OndePlane-----###
# Generation ondes planes et sommes
# auteur : Panayotis Akridas , fridjine ibrahim dublanc matthieu et ikram
# mail : pakridas@cyu.fr
# contributeur : Claude de Anthropic AI pour consolider le code et ameliorer la representation graphique (9/5/2026 version non payante)
# date creation : 4 mai 2026
# version 4 : 10 mai 2026
###


import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
from matplotlib import pyplot
import matplotlib.pyplot as plt

DEUX_PI = 2 * numpy.pi


def Compute_length(x: float | numpy.ndarray | list) -> int:
    """Determine le nombre d'elements dans la variable x (1 si scalaire)."""
    if isinstance(x, (numpy.ndarray, list)):
        return len(x)
    return 1


def Get_simulation_type(x, t) -> str:
    """Retourne le type de simulation selon la longueur de x et t."""
    len_x = Compute_length(x)
    len_t = Compute_length(t)
    if len_x > 1 and len_t == 1:
        msg = "espace"
    elif len_x == 1 and len_t > 1:
        msg = "temps"
    elif len_x == 1 and len_t > 1:
        msg = "ici_maintenant mais sans interet"
    else:
        print("")
        print("         Desole JARVIS, mais je ne sais pas encore faire !")
        print("")
        exit()
    return msg


def Check_momentum(k: float) -> float:
    """Verifie que le nombre d'onde est positif."""
    return abs(k)


def Compute_wavelength(k: float) -> float:
    """Calcule la longueur d'onde : lambda = 2*pi/k."""
    k = Check_momentum(k)
    return DEUX_PI / k


def Compute_omega(k: float, speed: float = 1.0, dispersion_type: str = "linear") -> float:
    """
    Calcule la pulsation omega a partir de la relation de dispersion.
    """
    k = Check_momentum(k)
    if dispersion_type == "linear":
        return speed * k
    elif dispersion_type == "schrodinger":
        hbar = 1.0546e-34   # [hbar] = M.L^2.T^{-1}
        masse = 9.109e-31   # [masse] = M
        return hbar * k**2 / (2 * masse)
    else:
        raise ValueError(f"Type de dispersion inconnu : '{dispersion_type}'")


def PlaneWave(amp, k, omega, x, t) -> numpy.ndarray:
    """
    Onde plane complexe 1D : Psi(x,t) = amp * exp(i(kx - omega*t))
    Fonction demandee par le sujet (question 1.1.a).

    numpy.ndarray : amplitude complexe de l'onde plane
    """
    x = numpy.asarray(x)
    return amp * numpy.exp(1j * (k * x - omega * t))


def Compute_plane_wave(k: float, x, t, dispersion_type: str = "linear") -> numpy.ndarray:
    """
    Calcule une onde plane complexe en 1d avec calcul automatique de omega.

    Parametres
    ----------
    k : float
        Nombre d'onde [1/m].
    x : array-like
        Positions spatiales.
    t : float ou array-like
        Instant(s) considere(s).
    dispersion_type : str
        Type de relation de dispersion.

    Retourne
    --------
    numpy.ndarray : amplitude complexe de l'onde plane
    """
    k = Check_momentum(k)
    omega = Compute_omega(k, dispersion_type=dispersion_type)
    x = numpy.asarray(x)
    t = numpy.asarray(t)
    phase = 1j * (omega * t - k * x)
    return numpy.exp(phase)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def Check_oscillations(k: float, x, t) -> tuple[float, str]:
    """
    Compte le nombre d'oscillations visibles sur l'intervalle et avertit
    si ce nombre est trop faible ou trop eleve.

    Retourne
    --------
    (n_oscillations, message)
    """
    str_genre = Get_simulation_type(x, t)

    if str_genre == "espace":
        lam = Compute_wavelength(k)
        min_intervalle = min(x)
        max_intervalle = max(x)
    if str_genre == "temps":
        lam = Compute_period(k)
        min_intervalle = min(t)
        max_intervalle = max(t)

    print(str_genre)
    etendue = abs(max_intervalle - min_intervalle)

    n_osc = etendue / lam
    if etendue < lam:
        msg = "/!\ Moins d'une oscillation visible -> agrandir l'intervalle ?"
    elif etendue > 6 * lam:
        msg = "/!\ Plus de 6 oscillations visibles -> reduire l'intervalle ?"
    else:
        msg = ""

    return n_osc, msg


def Compute_period(k):
    """Calcule la periode temporelle T = 2*pi/omega."""
    omega = Compute_omega(k)
    periode = DEUX_PI / omega
    return periode


def Compute_n_pts(k: float, facteur: int = 500) -> int:
    """
    Suggere un nombre de points pour un trace agreable.

    Parametres
    ----------
    k : float
        Nombre d'onde [1/m].
    facteur : int
        Nombre de points par longueur d'onde.
    """
    return int(facteur * Compute_wavelength(k))


def Get_info(liste_k: list, x, t, n_ondes: int) -> list[str]:
    """Construit un resume lisible des parametres de la simulation."""

    min_k = min(liste_k)
    max_k = max(liste_k)

    str_genre = Get_simulation_type(x, t)

    if str_genre == "espace":
        str_variable = "x"
        min_intervalle = min(x)
        max_intervalle = max(x)
        n_pts = Compute_length(x)

    if str_genre == "temps":
        str_variable = "t"
        min_intervalle = min(t)
        max_intervalle = max(t)
        n_pts = Compute_length(t)

    print(str_variable)
    lignes = [
        "",
        f"Simulation de {n_ondes} ondes planes :",
        f" - intervalle {str_variable} : {str_genre} [{min_intervalle:.3g}, {max_intervalle:.3g}]",
        f" - avec {n_pts} points",
        f" - intervalle nombres d'ondes k : [{min_k:.4g}, {max_k:.4g}]",
        f" - instant t = {t}",
        ""]
    return lignes


# ---------------------------------------------------------------------------
# Representation graphique
# ---------------------------------------------------------------------------

def Plot_re_im(k0=1.0, t=0.0):
    """
    Represente Re(Psi) et Im(Psi) en fonction de x a l'instant t.
    Correspond a la question 1.1.b du sujet.
    Utilise la syntaxe exacte demandee : fig, ax = plt.subplots() et ax.plot()
    """
    omega = Compute_omega(k0)
    # Intervalle : 3 longueurs d'onde
    x_trace = linspace(0, 3 * DEUX_PI / k0, 1000)

    psi = PlaneWave(amp=1.0, k=k0, omega=omega, x=x_trace, t=t)

    fig, ax = plt.subplots()
    ax.plot(x_trace, real(psi), label=r"Re[$\Psi(x,t)$]")
    ax.plot(x_trace, imag(psi), label=r"Im[$\Psi(x,t)$]", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi(x,t)$")
    ax.set_title(f"Onde plane — k={k0}, t={t}")
    ax.legend()
    plt.show()


def Plot_waves(x, t, ondes: numpy.ndarray, liste_k: list, liste_n_osc: list) -> None:
    """
    Trace chaque onde individuelle (partie reelle) et leur superposition.
    Trace egalement l'enveloppe analytique pour 3 ondes (question 1.2.1.d).

    Parametres
    ----------
    x : array
        Grille spatiale.
    t : float
        Instant considere.
    ondes : ndarray, taille (n_ondes, n_pts)
        Amplitudes complexes des ondes.
    liste_k : list
        Nombres d'onde (un par ligne dans ondes).
    liste_n_osc : list
        Nombre d'oscillations par onde (pour les legendes).
    """
    genre = Get_simulation_type(x, t)
    if genre == 'espace':
        intervalle = x
    if genre == 'temps':
        intervalle = t

    n_ondes = ondes.shape[0]
    fig, axes = pyplot.subplots(n_ondes + 1, 1,
                                figsize=(10, 2.5 * (n_ondes + 1)),
                                sharex=True)
    fig.suptitle("Superposition d'ondes planes", fontsize=14)

    couleurs = pyplot.cm.viridis(numpy.linspace(0.1, 0.85, n_ondes))

    for i, ax in enumerate(axes[:-1]):
        ax.plot(intervalle, numpy.real(ondes[i]), color=couleurs[i], linewidth=1.2)
        lam = Compute_wavelength(liste_k[i])
        ax.set_ylabel(f"onde {i}\n"
                      f"k={liste_k[i]:.3g}, λ={lam:.2f}\n"
                      f"n_osc={liste_n_osc[i]:.1f}",
                      fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    superposition = numpy.real(numpy.sum(ondes, axis=0))
    axes[-1].plot(intervalle, superposition, color="crimson", linewidth=1.5)

    # Enveloppe analytique (question 1.2.1.d) : |1 + cos(delta_k/2 * x)|
    # Valable pour 3 ondes : k0-delta_k/2, k0, k0+delta_k/2 avec amplitudes 0.5, 1, 0.5
    if len(liste_k) == 3:
        delta_k = abs(liste_k[2] - liste_k[0])
        enveloppe = numpy.abs(1 + numpy.cos(delta_k / 2 * intervalle))
        axes[-1].plot(intervalle,  enveloppe, color="black", lw=1.2, ls="--", label="enveloppe")
        axes[-1].plot(intervalle, -enveloppe, color="black", lw=1.2, ls="--")
        axes[-1].legend(fontsize=8)

    axes[-1].set_ylabel("superposition", fontsize=8)
    axes[-1].set_xlabel("x [m]")
    axes[-1].tick_params(labelsize=8)
    axes[-1].grid(True, linewidth=0.4, alpha=0.5)

    pyplot.tight_layout()
    pyplot.show()


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Parametres des nombres d'onde -----------------------------------
    k_centre = 1.0
    delta_k  = k_centre / 10

    # --- Grille spatiale : intervalle [-pi/delta_k, pi/delta_k] (sujet 1.2.1.d) ---
    x_ini = -numpy.pi / delta_k
    x_fin =  numpy.pi / delta_k
    n_pts = Compute_n_pts(k_centre)
    x = numpy.linspace(x_ini, x_fin, n_pts)

    # --- Instant initial -------------------------------------------------
    t = 0.0

    # --- 3 ondes du sujet (question 1.2.1.b) : k0-delta_k/2, k0, k0+delta_k/2 ---
    liste_k = [
        k_centre - delta_k / 2,
        k_centre,
        k_centre + delta_k / 2,
    ]
    # Amplitudes : 1 pour k0, 1/2 pour les deux ondes laterales (enonce 1.2.1.b)
    liste_amp = [0.5, 1.0, 0.5]
    n_ondes = 3

    # --- Affichage du type de simulation ---------------------------------
    print(Get_simulation_type(x, t))

    # --- Affichage du resume de la simulation ----------------------------
    for ligne in Get_info(liste_k[:n_ondes], x, t, n_ondes):
        print(ligne)

    # --- Calcul des ondes ------------------------------------------------
    # Tableau de zeros de taille n_ondes x n_pts
    ondes = numpy.zeros((n_ondes, n_pts), dtype=complex)
    liste_n_osc: list[float] = []
    liste_msg:   list[str]   = []

    for i in range(n_ondes):
        k_i     = liste_k[i]
        amp_i   = liste_amp[i]
        omega_i = Compute_omega(k_i)
        ondes[i] = PlaneWave(amp_i, k_i, omega_i, x, t)
        n_osc, msg = Check_oscillations(k_i, x, t)
        liste_n_osc.append(n_osc)
        liste_msg.append(msg)

    # --- Affichage du rapport par onde -----------------------------------
    for i in range(n_ondes):
        lam = Compute_wavelength(liste_k[i])
        ligne = (f"Onde n°{i}  lambda={lam:.2f} m  "
                 f"n_osc={liste_n_osc[i]:.2f}")
        if liste_msg[i]:
            ligne += f"  {liste_msg[i]}"
        print(ligne)

    # --- 1.1.b : trace Re et Im de l'onde centrale ----------------------
    Plot_re_im(k0=k_centre, t=t)

    # --- 1.2 : trace superposition + enveloppe --------------------------
    Plot_waves(x, t, ondes, liste_k[:n_ondes], liste_n_osc)
