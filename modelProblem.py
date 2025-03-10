import math

import natConstSI


class ModelProblem:
    """
    Container for the parameters specifying a radiative transfer problem:

        - in one or two spatial dimension
        - for radation of a single frequency
        - in a possibly inhomogeneous medium, made up of a single material of
          constant temperature
        - with isotropic scattering

    Attributes
    ----------
    alb : float
        Albedo of the medium.
    emiss : float
        Emissivity of the medium
    scat : string
        Type of scattering process assumed for the medium
    abs_fun : callable
        The spatially varying absorption coefficient in a domain that is filled
        with a single material of spatially varying density. This implies the
        ratio between scattering coefficient and absorption coefficient is at a
        fixed value xi throughout the domain.
    xi : float
        The ratio between scattering coefficient and absorption coefficient:
        alpha_scat = xi * alpha_abs.
    xip1 : float
        The value of xi + 1.0, stored for conveniency.
    s_e : float
        Value of the dimensionless planck function for given frequency of
        radiation and temperature of medium.
    """

    def __init__(self, temperature, frequency, albedo, emissivity,
                 scattering, absorption_fun):
        """
        Parameters
        ----------
        temperature : float
            Temperature of the medium, measured in Kelvin. It is assumed
            to be constant on the timescale of interest.
        frequency : float
            Frequency of relevant radiation.
        albedo : float, 0 <= alb <= 1
            Albedo of the medium, i.e. a measure of the relative strength of
            scattering compared to absorption.
        emissivity : float, 0 <= emiss <= 1
            Emissivity of the medium.
        scattering : string
            Type of scattering process assumed for the medium
        absorption_fun : callable
            Absorption coefficient assumed for the medium.
        """
        self.s_e = 0

        e_ratio = (natConstSI.h_pla * frequency) / \
                  (natConstSI.k_bol * temperature)

        self.s_e = 1.0 / math.expm1(e_ratio)

        assert 0.0 <= albedo and albedo < 1.0, \
            'Invalid albedo value. Must be in [0,1).'
        self.alb = albedo

        assert 0.0 <= emissivity and emissivity <= 1.0, \
            'Invalid emissivity value. Must be in [0,1].'
        self.emiss = emissivity

        self.xi = 1.0 / (1.0 - albedo) - 1.0
        self.xip1 = self.xi + 1.0

        assert scattering in ['none', 'isotropic'], \
            'Scattering process "' + scattering + '" not implemented.'
        self.scat = scattering

        assert callable(absorption_fun), \
            'The absorption coefficient must be callable and take a value' + \
            ' in the domain as argument.'
        self.abs_fun = absorption_fun

        print('\n\nModel problem:\n' +
              '--------------\n' +
              '    - temperature: ' + str(temperature) + ' K\n' +
              '    - frequency: ' + str(frequency/1e12) + ' THz\n' +
              '    - s_e: ' + str(self.s_e) + '\n' +
              '    - albedo: ' + str(albedo) + '\n' +
              '    - emissivity: ' + str(emissivity) + '\n' +
              '    - scattering: ' + scattering + '\n\n')
