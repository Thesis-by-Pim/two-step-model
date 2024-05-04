import time
from mip import maximize, xsum
from linear_programming_setup.mipSolver import MIPSolver
from linear_programming_setup.problemDef import ProblemDef
from loguru import logger as log

class MaxValueSolver(MIPSolver):
    """
    This class models and solves a garden planting plan using MILP
    """

    # Main solver method for the ILP
    def _solve(self) -> ProblemDef:
        log.inf("Building model")
        self._buildModel(name=str(type(self)))
        self._setGurobiVerbosity()
        
        self.problem.timegranularity = 1
        
        self._addCropTypeOverlappingContstraints()
        self._addOneDimensionalPieceWiseAreaConstraint()

        self.model.objective = maximize(
            xsum(self.problem.get_market_value(i) * self._cropCounts[i] for i in range(self.problem.get_crop_count()))
        )

        if self.problem.timeoutValueAfter == 0:
            maxSeconds = 10
        else:
            maxSeconds = self.problem.timeoutValueAfter

        status = self.model.optimize(max_seconds=maxSeconds)
        
        log.trace("sleep")
        time.sleep(1)
            
        self._setResults(status=status)        
        
        return self.problem