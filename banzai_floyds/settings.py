from banzai.settings import *  # noqa: F401,F403

ORDERED_STAGES = [
                  'banzai.bias.OverscanSubtractor',
                  ]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]

SUPPORTED_FRAME_TYPES = ['SPECTRUM', 'LAMPFLAT', 'ARC']

LAST_STAGE = {'SPECTRUM': None, 'LAMPFLAT': None, 'ARC': None}

EXTRA_STAGES = {'SPECTRUM': None, 'LAMPFLAT': None, 'ARC': None}
