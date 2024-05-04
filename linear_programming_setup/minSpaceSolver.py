import json
import time
from mip import minimize, xsum
from linear_programming_setup.mipSolver import MIPSolver
from linear_programming_setup.problemDef import ProblemDef
from loguru import logger as log

class MinSpaceSolver(MIPSolver):
    """
    This class models and solves a garden planting plan using MILP
    """

    # Main solver method for the ILP
    def _solve(self) -> ProblemDef:
        start_time = time.process_time()
        rerunning = True
        
        measure1 = time.process_time() - start_time

        log.info("Building model")

        self._buildModel(name="Nutrition_minspace") 
        self._setGurobiVerbosity()
        
        self._addNutritionConstraints()
        self._addCropTypeOverlappingContstraints()
        self._addOneDimensionalPieceWiseAreaConstraint()
        
        log.info("Model building complete")

        self.model.objective = minimize(
            xsum(self.problem.get_crop_resource_requirement(j,0) * self._cropCounts[j] for j in range(self.problem.get_crop_count()))
        )

        measure3 = time.process_time() - measure1

        if self.problem.timeoutValueAfter == 0:
            maxSeconds = 20
        else:
            maxSeconds = self.problem.timeoutValueAfter
        
        status = self.model.optimize(max_seconds=maxSeconds)
        
        self._setResults(status)

        print(json.dumps(self.problem.print_results(), indent=4))
        self.problem.plot_results_2D()

        return self.problem