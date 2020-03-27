import math

import natConstSI


class ModelProblem1d:
    """
    Container for the parameters specifying a radiative transfer problem:

        - in one spatial dimension
        - for radation of a single frequency
        - in a possibly inhomogeneous medium, made up of a single material of
          constant temperature
        - with isotropic scattering

    Attributes
    ----------
    alb : float
        Albedo of the medium.
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
    dom_len : float
        Length of the one-dimensional domain.
    inflow_bc : tuple of length 2
        Boundary conditions for the inflow boundaries of the corresponding
        discrete ordinates.
    s_eps : float
        Value of the dimensionless planck function for given frequency of
        radiation and temperature of medium.
    """

    def __init__(self, temperature, frequency, albedo, scattering,
                 absorption_fun, domain_len, inflow_bc):
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
        scattering : string
            Type of scattering process assumed for the medium
        absorption_fun : callable
            Absorption coefficient assumed for the medium.
        domain_len : float
            Length of the one-dimensional domain. The domain itself is then
            defined as D = (0, dom_len).
        inflow_bc : tuple of length 2
            Boundary conditions for the inflow boundaries of the corresponding
            discrete ordinates. In one dimension there are exactly two
            ordinates.
        """

        self.s_eps = 0

        e_ratio = (natConstSI.h_pla * frequency) / \
                  (natConstSI.k_bol * temperature)

        self.s_eps = 1.0 / math.expm1(e_ratio)

        assert 0.0 <= albedo and albedo < 1.0, \
            'Invalid albedo value. Must be in [0,1).'
        self.alb = albedo
        self.xi = 1.0 / (1.0 - albedo) - 1.0
        self.xip1 = self.xi + 1.0

        assert scattering in ['none', 'isotropic'], \
            'Scattering process "' + scattering + '" not implemented.'
        self.scat = scattering

        assert callable(absorption_fun), \
            'The absorption coefficient must be callable and take a value' + \
            ' in the domain as argument.'
        self.abs_fun = absorption_fun

        assert domain_len > 0, 'Invalid domain length. Must be positive.'
        self.dom_len = domain_len

        assert len(inflow_bc) == 2, \
            'Invalid inflow boundary conditions.' + \
            'For a 1d problem there must be exactly 2 conditions.'
        self.inflow_bc = inflow_bc

        print('\n\nModel problem:\n' +
              '--------------\n' +
              '    - dimension: 1\n' +
              '    - domain: (0,' + str(domain_len) + ')\n' +
              '    - temperature: ' + str(temperature) + ' K\n' +
              '    - frequency: ' + str(frequency/1e12) + ' THz\n' +
              '    - s_eps: ' + str(self.s_eps) + '\n' +
              '    - albedo: ' + str(albedo) + '\n' +
              '    - isotropic scattering\n\n\n')
