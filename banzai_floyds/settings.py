from banzai.settings import *  # noqa: F401,F403

ORDERED_STAGES = [
                  'banzai.bias.OverscanSubtractor',
                  'banzai.trim.Trimmer',
                  'banzai_floyds.orders.OrderLoader'
                  ]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]

SUPPORTED_FRAME_TYPES = ['SPECTRUM', 'LAMPFLAT', 'ARC', 'SKYFLAT']

LAST_STAGE = {'SPECTRUM': None, 'LAMPFLAT': None, 'ARC': None, 'SKYFLAT': None}

EXTRA_STAGES = {'SPECTRUM': None, 'LAMPFLAT': None, 'ARC': None, 'SKYFLAT': ['banzai_floyds.orders.OrderSolver']}

FRAME_FACTORY = 'banzai_floyds.frames.FLOYDSFrameFactory'
