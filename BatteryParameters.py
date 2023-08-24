"""
CIF Project 2022

Physics-Informed Neural Networks for Next-Generation Aircraft

Battery Parameters

Matteo Corbetta
matteo.corbetta@nasa.gov
"""

def default():
      params = {
            'xnMax': 0.6,         # Mole fractions on negative electrode (max)
            'xnMin': 0.,          # Mole fractions on negative electrode (min)
            'xpMax': 0.6,         # Mole fractions on positive electrode (max)
            'xpMin': 0.,          # Mole fractions on positive electrode (min)
            'Ro': 0.117215,       # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
            'R':  8.3144621,      # universal gas constant, J/K/mol
            'F':  96487,          # Faraday's constant, C/mol
            'alpha': 0.5,         # anodic/cathodic electrochemical transfer coefficient
            'Sn': 0.000437545,    # surface area (- electrode)
            'Sp': 0.00030962,     # surface area (+ electrode)
            'kn': 2120.96,        # lumped constant for BV (- electrode)
            'kp': 248898,         # lumped constant for BV (+ electrode)
            'Volume': 2e-5,       # half interior battery volume/2 (for computing concentrations)
            'VolumeSurf': 0.1,    # fraction of total volume occupied by surface volume
            'qMobile': 7600,     
            'tDiffusion': 7e6,    # diffusion time constant (increasing this causes decrease in diffusion rate)
            'to':  6.08671,       # for Ohmic voltage
            'tsn': 1001.38,       # for surface overpotential (neg)
            'tsp': 46.4311,       # for surface overpotential (pos) 
      }
      params['qmax'] = params['xnMax'] - params['xnMin']
      return  params


def rkexp_default():
      params_p = {'U0': 4.03, 'A0':  -31593.7, 'A1':  0.106747, 'A2':  24606.4,  'A3':  -78561.9,
                              'A4':  13317.9,  'A5':  307387.0, 'A6':  84916.1,  'A7':  -1.07469e+06,
                              'A8':  2285.04,  'A9':  990894.0, 'A10': 283920.0, 'A11': -161513.0, 'A12': -469218.0}
      params_n = {'U0': 0.01, 'As':  [86.19, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]}
      return {'positive': params_p, 'negative': params_n}


if __name__ == '__main__':
    
    print("Battery Parameters Script.")