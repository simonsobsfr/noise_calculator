import numpy as np
import pylab as plt
import kairos_noise_calculator as so_noise

def read_ps(file_name, spectra=None):
    """Read the power spectra.
        
    Parameters
    ----------
    file_name: str
      the name of the file to read the spectra
    spectra: list of strings
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
      
    Return
    ----------
      
    The function return the multipole l (or binned multipoles) and a 1d power spectrum
    array or a dictionnary of power spectra (if spectra is not None).
    """
    
    data = np.loadtxt(file_name)
    if spectra is None:
        return data[:, 0], data[:, 1]

    l = data[:, 0]
    ps = {spec: data[:, i + 1] for i, spec in enumerate(spectra)}
    return l, ps
    
    
spectra = ['TT', 'TE', 'TB', 'ET', 'BT', 'EE', 'EB', 'BE', 'BB']
l, ps = read_ps("cmb.dat", spectra=spectra)

sensitivity_mode = 1
one_over_f_mode = 1
f_sky = 0.4
ell_min = 2
ell_max = 500
delta_ell = 1
n_years = 2
NTubes_LF = 0
NTubes_MF = 1
NTubes_UHF = 0
obs_eff = 0.2

freq_list = [27, 39, 93, 145, 225, 280]

ell, N_ell_P_SA, Map_white_noise_levels = so_noise.Simons_Observatory_V3_SA_noise(sensitivity_mode,
                                                                                  one_over_f_mode,
                                                                                  f_sky,
                                                                                  ell_min,
                                                                                  ell_max,
                                                                                  delta_ell,
                                                                                  n_years,
                                                                                  NTubes_LF,
                                                                                  NTubes_MF,
                                                                                  NTubes_UHF,
                                                                                  obs_eff = obs_eff)

fac = ell * (ell + 1) / ( 2* np.pi)

plt.loglog()
plt.xlim(0, ell_max)
plt.plot(l, ps["EE"])
plt.plot(l, ps["BB"])
for i, freq in enumerate(freq_list):
    print(freq, Map_white_noise_levels[i])
    
    if freq in [93, 145]:
        plt.plot(ell, N_ell_P_SA[i] * fac, label=f"{freq} GHz")
plt.legend()
plt.savefig("noise.png")
plt.clf()
plt.close()

