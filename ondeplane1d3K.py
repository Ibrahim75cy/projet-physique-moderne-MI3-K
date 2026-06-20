#OndePlane


import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
from matplotlib import pyplot
import matplotlib.pyplot as plt

DEUX_PI = 2 * numpy.pi

# Renvoie le nombre d'elements de x (1 si c'est un scalaire)
def Compute_length(x: float | numpy.ndarray | list) -> int:
    if isinstance(x, (numpy.ndarray, list)):
        return len(x)
    return 1

# Determine si on simule selon l'espace ou selon le temps (selon ce qui varie)
def Get_simulation_type(x, t) -> str:
    len_x = Compute_length(x)
    len_t = Compute_length(t)
    if len_x > 1 and len_t == 1:
        msg = "espace"
    elif len_x == 1 and len_t > 1:
        msg = "temps"
    else:
        print("")
        print("         Desole JARVIS, mais je ne sais pas encore faire !")
        print("")
        exit()
    return msg

# Force un nombre d'onde positif
def Check_momentum(k: float) -> float:
    return abs(k)

# Longueur d'onde lambda = 2*pi/k
def Compute_wavelength(k: float) -> float:
    k = Check_momentum(k)
    return DEUX_PI / k

# Pulsation omega selon la relation de dispersion (lineaire ou Schrodinger)
def Compute_omega(k: float, speed: float = 1.0, dispersion_type: str = "linear") -> float:
    k = Check_momentum(k)
    if dispersion_type == "linear":
        return speed * k
    elif dispersion_type == "schrodinger":
        hbar = 1.0546e-34   
        masse = 9.109e-31   
        return hbar * k**2 / (2 * masse)
    else:
        raise ValueError(f"Type de dispersion inconnu : '{dispersion_type}'")

# Onde plane complexe 1D : Psi = amp * exp(i(kx - omega*t))  
def PlaneWave(amp, k, omega, x, t) -> numpy.ndarray:
    x = numpy.asarray(x)
    return amp * numpy.exp(1j * (k * x - omega * t))

# Onde plane avec calcul automatique de omega a partir de la dispersion
def Compute_plane_wave(k: float, x, t, dispersion_type: str = "linear") -> numpy.ndarray:
    k = Check_momentum(k)
    omega = Compute_omega(k, dispersion_type=dispersion_type)
    x = numpy.asarray(x)
    t = numpy.asarray(t)
    phase = 1j * (omega * t - k * x)
    return numpy.exp(phase)

# Compte les oscillations visibles et avertit si l'intervalle est trop petit/grand
def Check_oscillations(k: float, x, t) -> tuple[float, str]:
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

# Periode temporelle T = 2*pi/omega
def Compute_period(k):
    omega = Compute_omega(k)
    periode = DEUX_PI / omega
    return periode

# Suggere un nombre de points pour un trace lisible (facteur points par longueur d'onde)
def Compute_n_pts(k: float, facteur: int = 500) -> int:
    return int(facteur * Compute_wavelength(k))

# Construit un resume texte des parametres de la simulation
def Get_info(liste_k: list, x, t, n_ondes: int) -> list[str]:

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

# Representation graphique

# Trace Re(Psi) et Im(Psi) d'une onde plane en fonction de x 
def Plot_re_im(k0=1.0, t=0.0):
    omega = Compute_omega(k0)
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

# Trace les ondes individuelles, leur somme et l'enveloppe 
def Plot_waves(x, t, ondes: numpy.ndarray, liste_k: list, liste_n_osc: list) -> None:
    genre = Get_simulation_type(x, t)
    if genre == 'espace':
        intervalle = x
    if genre == 'temps':
        intervalle = t

    n_ondes = ondes.shape[0]
    couleurs = pyplot.cm.viridis(numpy.linspace(0.1, 0.85, n_ondes))

    fig, ax = pyplot.subplots(figsize=(10, 5))
    fig.suptitle("Superposition d'ondes planes", fontsize=14)

    for i in range(n_ondes):
        lam = Compute_wavelength(liste_k[i])
        ax.plot(intervalle, numpy.real(ondes[i]),
                color=couleurs[i], linewidth=1.2,
                label=f"Re[onde {i}]  k={liste_k[i]:.3g}, λ={lam:.2f}")

    superposition = numpy.real(numpy.sum(ondes, axis=0))
    ax.plot(intervalle, superposition,
            color="crimson", linewidth=2.0, label="Re[somme]")

    # Enveloppe analytique |1 + cos(delta_k/2 * x)| (valable pour 3 ondes 0.5/1/0.5)
    if len(liste_k) == 3:
        delta_k = abs(liste_k[2] - liste_k[0])
        enveloppe = numpy.abs(1 + numpy.cos(delta_k / 2 * intervalle))
        ax.plot(intervalle,  enveloppe, color="black", lw=1.5, ls="--", label="enveloppe")
        ax.plot(intervalle, -enveloppe, color="black", lw=1.5, ls="--")

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    pyplot.tight_layout()
    pyplot.show()


#Main

if __name__ == "__main__":

    # Parametres des nombres d'onde
    k_centre = 1.0
    delta_k  = k_centre / 10

    # Grille spatiale [-pi/delta_k, pi/delta_k] 
    x_ini = -numpy.pi / delta_k
    x_fin =  numpy.pi / delta_k
    n_pts = Compute_n_pts(k_centre)
    x = numpy.linspace(x_ini, x_fin, n_pts)

    # Instant initial 
    t = 0.0

    # 3 ondes du sujet : k0-delta_k/2, k0, k0+delta_k/2 
    liste_k = [
        k_centre - delta_k / 2,
        k_centre,
        k_centre + delta_k / 2,
    ]
    
    liste_amp = [0.5, 1.0, 0.5] # amplitude 1 pour k0, 1/2 pour les laterales
    n_ondes = 3

    print(Get_simulation_type(x, t))

    for ligne in Get_info(liste_k[:n_ondes], x, t, n_ondes):
        print(ligne)

    # Calcul des ondes 
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

    # Rapport par onde 
    for i in range(n_ondes):
        lam = Compute_wavelength(liste_k[i])
        ligne = (f"Onde n°{i}  lambda={lam:.2f} m  "
                 f"n_osc={liste_n_osc[i]:.2f}")
        if liste_msg[i]:
            ligne += f"  {liste_msg[i]}"
        print(ligne)

    # Trace Re et Im de l'onde centrale 
    Plot_re_im(k0=k_centre, t=t)

    # Trace superposition + enveloppe 
    Plot_waves(x, t, ondes, liste_k[:n_ondes], liste_n_osc)
