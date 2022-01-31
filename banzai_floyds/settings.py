from banzai.settings import *  # noqa: F401,F403

ORDERED_STAGES = [
                  'banzai.bias.OverscanSubtractor',
                  ]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]
