from rl4co.utils.pylogger import get_pylogger

from ..pctsp.env import PCTSPEnv

log = get_pylogger(__name__)


class SPCTSPEnv(PCTSPEnv):
    """Stochastic Prize Collecting Traveling Salesman Problem (SPCTSP) environment.

    Note:
        The only difference with deterministic PCTSP is that the prizes are stochastic
        (i.e. the expected prize is not the same as the real prize).
    """

    name = "spctsp"
    _stochastic = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is False:
            log.warning(
                "Deterministic mode should not be used for SPCTSP. Use PCTSP instead."
            )
