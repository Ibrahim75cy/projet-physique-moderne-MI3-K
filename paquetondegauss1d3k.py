#PaquetOndesGaussien

import numpy
from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import scipy

# Parametres du paquet d_ondes initial
K0    = 1   
SIGMA = 0.2   
X0    = -20   

# Constantes physiques en unites reduites (hbar = m = 1)
HBAR  = 1
HBAR2 = HBAR*HBAR 
MASS  = 1 
HALF = .5

# Constantes demandees par le sujet
hbar = HBAR   
m    = MASS   

# Grille d_espace
X_MIN, X_MAX = -20, 100
N_X          = 2000
x            = numpy.linspace(X_MIN, X_MAX, N_X)

# Gaussienne normalisee (via scipy) ; ne marche pas avec des complexes
def Compute_re_norm_gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

# Gaussienne non normalisee ; fonctionne avec des arguments complexes
def Compute_gaussian(x, mu, sigma):
    return numpy.exp(-(x-mu)**2/(2 * sigma**2))

# Vitesse de phase v_phi = hbar k0 / (2m)
def Compute_phase_velocity(k0 = K0):
    return HBAR*k0/(2*MASS)

# Vitesse de groupe v_g = hbar k0 / m = 2 v_phi
def Compute_group_velocity(k0 = K0):
    return Compute_phase_velocity(2*k0)

# Integre |psi|^2 (Simpson) : doit valoir 1 pour une densite de probabilite
def Check_normalization(function_array, x_interval):
    x_min = min(x_interval)
    x_max = max(x_interval)
    return scipy.integrate.simpson(function_array, x_interval)

# Paquet gaussien au cours du temps ; non utilise ici
def Compute_gaussian_wp_as_cct(x, t, sigma=SIGMA, k0=K0, x0=X0):
    a = 1/sigma
    a2 = a**2
    terme = 2*1j*HBAR*t/MASS
    amplitude = numpy.power(2*a2/(pi*(a2**2 + 4*HBAR2*t**2/(MASS**2))),.25)
    mu = HBAR*k0*t/MASS
    new_sigma = numpy.sqrt((a2+2*1j*HBAR*t/MASS)/2)
    return amplitude * numpy.exp(1j*k0*x) * Compute_gaussian(x, mu, new_sigma)

# Paquet gaussien au cours du temps : c'est celui qu'on utilise
def Compute_gaussian_wp(x, t, sigma=SIGMA, k0=K0, x0=X0):
    a = 1/sigma
    a2 = a**2
    terme = MASS*a2+2*1j*HBAR*t
    inv_terme = 1/terme
    amplitude = numpy.sqrt(2*MASS*a*inv_terme/(numpy.sqrt(2*pi)))
    return amplitude * numpy.exp(inv_terme*MASS*(a2*k0+2*1j*x)**2/4) * Compute_gaussian(k0,0,numpy.sqrt(2)/a)


# Densite initiale du paquet ; verification/debug
def Compute_ini_gaussian2_wp_as_cct(x, sigma=SIGMA, k0=K0):
    sigma2 = sigma**2
    phase = numpy.exp(1j*k0*x)
    arg_exp = -sigma2*x**2    
    amplitude = numpy.power(2*sigma2/pi,0.25)
    return amplitude*phase*numpy.exp(arg_exp)

# Position theorique du centre du paquet : <x> = v_g * t
def Compute_analytic_gaussian_position(t, k0 = K0):
    return Compute_group_velocity(k0)*t

# Dispersion theorique du paquet en fonction du temps
def Compute_analytic_gaussian_spreading(t, sigma=SIGMA):
    return HALF/sigma * numpy.sqrt(1+(2*HBAR*t*sigma**2/MASS)**2)

# Hauteur theorique du pic de densite en fonction du temps
def Compute_analytic_gaussian_height(t, sigma=SIGMA):
    a=1/sigma
    a2 = a**2
    return numpy.sqrt(2/(pi*a2*(1+(2*HBAR*t/(MASS*a2))**2)))

# Hauteur numerique (max de la densite)
def Compute_numeric_height(array_function):
    return max(array_function)

# Position numerique du maximum de la densite
def Compute_numeric_position(array_function, x):
    index_max = numpy.argmax(array_function) 
    return x[index_max]

# Moment d'ordre donne de la densite (sert au calcul de la dispersion)
def Compute_numeric_moment(array_function, x, moment):
    array_moment = array_function*numpy.power(x, moment)
    return Check_normalization(array_moment, x)

# Dispersion numerique : ecart-type calcule a partir des moments
def Compute_numeric_spreading(array_function, x):
    mean_x  = Compute_numeric_moment(array_function, x, 1)
    mean_x2 = Compute_numeric_moment(array_function, x, 2)
    spreading = numpy.sqrt(mean_x2 - mean_x**2) 
    return spreading

# Paquet d'ondes gaussien Psi(x,t) 
def GaussWP(k0, a, x, t):
    x   = numpy.asarray(x, dtype=complex)
    a2  = a**2
    denominateur = m * a2 + 2j * hbar * t
    prefacteur   = numpy.sqrt(4 * pi * m * a / denominateur) * (1 / (8 * pi**3))**0.25
    exposant     = m / 4 * (a2 * k0 + 2j * x)**2 / denominateur - a2 * k0**2 / 4
    return prefacteur * numpy.exp(exposant)


# Trace Re(Psi) et Im(Psi) du paquet a t=0 
def Plot_re_im_wp(k0=K0, a=1/SIGMA, t=0.0):
    psi = GaussWP(k0=k0, a=a, x=x, t=t)

    fig, ax = plt.subplots()
    ax.plot(x, real(psi), label=r"Re[$\Psi(x,t=0)$]")
    ax.plot(x, imag(psi), label=r"Im[$\Psi(x,t=0)$]", linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi(x, 0)$")
    ax.set_title(f"Paquet d'ondes gaussien a t={t} — k0={k0}, a={a:.2f}")
    ax.legend()
    plt.show()


# Anime l'evolution du paquet : densite |psi|^2 (+ Re/Im selon 'kind'), norme, position, dispersion
def animate_wavepacket(t_max=30.0, n_frames=600, interval=10000000, kind=2):
    times = numpy.linspace(0, t_max, n_frames)

    #Calcul des donnees pour toutes les images
    print("Calcul des donnees pour toutes les images")
    all_norm      = numpy.empty(n_frames)
    all_position  = numpy.empty(n_frames)    
    all_spreading = numpy.empty(n_frames)
    all_height    = numpy.empty(n_frames)
    all_density   = numpy.empty((n_frames, len(x)))    

    if kind>1:
        all_re_psi = numpy.empty((n_frames, len(x)))
        all_im_psi = numpy.empty((n_frames, len(x)))


    for i, t_val in enumerate(times):
        psi         = Compute_gaussian_wp(x, t_val)
        density     = numpy.abs(psi)**2

        all_density[i]   = density
        all_norm[i]      = Check_normalization(density, x)
        all_position[i]  = Compute_analytic_gaussian_position(t_val)
        all_spreading[i] = Compute_analytic_gaussian_spreading(t_val)
        all_height[i]    = Compute_analytic_gaussian_height(t_val)

        if kind>1:
            all_re_psi[i] = numpy.real(psi)
            all_im_psi[i] = numpy.imag(psi)

    print(f"Done.  norm at t=0: {all_norm[0]:.6f},  at t={t_max}: {all_norm[-1]:.6f}")

    # Calcul des limites du graph
    y_max_density = all_density[0].max() * 1.5
    if kind>1:
        y_max_re = numpy.abs(all_re_psi[0]).max() * 1.5
        y_max_im = numpy.abs(all_re_psi[0]).max() * 1.5
    if kind == 1:
        fig, ax_density = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    if kind == 2:
        fig, (ax_density, ax_re) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if kind == 3:
        fig, (ax_density, ax_re, ax_im) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Evolution du paquet d_onde gaussien", fontsize=13)

    (line_density,) = ax_density.plot(x, all_density[0], color="#3266ad", lw=2)
    ax_density.set_ylim(0, y_max_density)
    ax_density.set_xlim(X_MIN, X_MAX)
    ax_density.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=11)
    time_text      = ax_density.text(0.02, 0.90, "temps t = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    norm_text      = ax_density.text(0.02, 0.76, f"norme (t) = {all_norm[0]:.4f}",
                         transform=ax_density.transAxes, fontsize=9,  color="gray")
    position_text  = ax_density.text(0.1, 0.90, "poistion du max : x0 = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    spreading_text = ax_density.text(0.3, 0.90, "dispersion : Delta x = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    height_text = ax_density.text(0.6, 0.90, "hauteur = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")

    # Sous-graphes Re et Im si demandes 
    if kind >1 :
        (line_re,) = ax_re.plot(x, all_re_psi[0], color="#1D9E75", lw=2)
        ax_re.axhline(0, color="gray", lw=0.5, ls="--")
        ax_re.set_ylim(-y_max_re, y_max_re)
        ax_re.set_xlim(X_MIN, X_MAX)
        ax_re.set_ylabel(r"Re$[\psi(x,t)]$", fontsize=11)
        ax_re.set_xlabel("x", fontsize=12)
        if kind >2:
            (line_im,) = ax_im.plot(x, all_re_psi[0], color="#1D9E75", lw=2)
            ax_im.axhline(0, color="gray", lw=0.5, ls="--")
            ax_im.set_ylim(-y_max_im, y_max_im)
            ax_im.set_xlim(X_MIN, X_MAX)
            ax_im.set_ylabel(r"Im$[\psi(x,t)]$", fontsize=11)
            ax_im.set_xlabel("x", fontsize=12)

    plt.tight_layout()

    # Mise a jour d'une image de l'animation
    def update(frame):
        line_density.set_ydata(all_density[frame])
        if kind > 1:
            line_re.set_ydata(all_re_psi[frame])
            if kind >2:
                line_im.set_ydata(all_im_psi[frame])
                
        time_text.set_text(f"t = {times[frame]:.2f}")
        norm_text.set_text(f"norme (t) = {all_norm[frame]:.4f}")
        position_text.set_text(f"poistion du max : x0 = {all_position[frame]:.4f}")
        spreading_text.set_text(f"dispersion : Delta x = {all_spreading[frame]:.4f}")
        height_text.set_text(f"hauteur = {all_height[frame]:.4f}")
        if kind == 1:
            return line_density, time_text, norm_text, position_text, spreading_text, height_text
        if kind == 2:    
            return line_density, line_re, time_text, norm_text, position_text, spreading_text
        if kind == 3:
            return line_density, line_re, line_im, time_text, norm_text, position_text, spreading_text
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True,
        repeat=True,
    )

    plt.show()

#Main
if __name__ == "__main__":

    # test de GaussWP (la norme a t=0 doit etre proche de 1)
    psi_test = GaussWP(k0=K0, a=1/SIGMA, x=x, t=0.0)
    densite_test = numpy.abs(psi_test)**2
    norme_test = scipy.integrate.simpson(densite_test, x)
    print(f"Test GaussWP — norme a t=0 : {norme_test:.6f} (doit etre proche de 1)")

    # trace Re et Im a t=0 
    Plot_re_im_wp(k0=K0, a=1/SIGMA, t=0.0)

    # Animation du paquet d'ondes 
    animate_wavepacket(t_max=40, n_frames=2000, interval=100, kind=1)
