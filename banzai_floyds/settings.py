from banzai.settings import *  # noqa: E401,E403

ORDERED_STAGES = [
                  'banzai.bias.OverscanSubtractor',
                  ]

FRAME_SELECTION_CRITERIA = [('type', 'contains', 'FLOYDS')]
