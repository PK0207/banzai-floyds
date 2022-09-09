from banzai.settings import *  # noqa: F401,F403

ORDERED_STAGES = ['banzai.bias.OverscanSubtractor',
                  'banzai.trim.Trimmer',
                  'banzai.gain.GainNormalizer',
                  'banzai.uncertainty.PoissonInitializer',
                  'banzai_floyds.orders.OrderLoader',
                  'banzai_floyds.wavelengths.WavelengthSolutionLoader']

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]

SUPPORTED_FRAME_TYPES = ['SPECTRUM', 'LAMPFLAT', 'ARC', 'SKYFLAT']

LAST_STAGE = {'SPECTRUM': None, 'LAMPFLAT': None, 'ARC': 'banzai_floyds.orders.OrderLoader',
              'SKYFLAT': 'banzai_floyds.orders.OrderLoader'}

EXTRA_STAGES = {'SPECTRUM': None, 'LAMPFLAT': None,
                'ARC': ['banzai_floyds.wavelengths.CalibrateWavelengths'],
                'SKYFLAT': ['banzai_floyds.orders.OrderSolver']}

FRAME_FACTORY = 'banzai_floyds.frames.FLOYDSFrameFactory'
