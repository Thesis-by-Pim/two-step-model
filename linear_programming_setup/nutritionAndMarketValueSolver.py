import time
from linear_programming_setup.mipSolver import MIPSolver
from mip import maximize, xsum, OptimizationStatus
from linear_programming_setup.problemDef import ProblemDef
from loguru import logger as log
from humanfriendly import format_timespan

class NutritionAndMarketValueSolver(MIPSolver):
    """
    This class models and solves a garden planting plan using MILP
    """

    # Main solver method for the ILP
    # @log.catch
    def _solve(self) -> ProblemDef:
        tg = self.problem.timegranularity
        
        # Run nutritional model to maximize nutrtition
        self._runNutritionModelTwoD()
        
        self.model.write("nutri.sol")
        
        # Check all nutri-targest succesfull
        notOptimalNutrition = sum(sum([1 - self._nutriTargetsSuccess[p][c].xi(0) for c in range(self.problem.get_nutrient_count())]) for p in range(tg)) > 0.002
        
        # Stopping conditions for after nutri opt when it did not reach full nutrition
        if notOptimalNutrition:
            self.problem.resultConclusion["Nutrition_maximization"] += " but Not achieved"
            log.warning("Full nutrition was not achieved")
            
            
            if self.problem.valueOptAfterNutri:
                if not self.problem.valueOptAfterNutriFailure:
                    log.opt(colors=True).info("<yellow>Skipping value opt</yellow>, since nutrition was not feasible")
                    log.inf("valueOptAfterNutriFailure=TRUE") 
                    return self.problem
        
        # Stopping after nutri opt if value opt is deactivated in configuration
        if not self.problem.valueOptAfterNutri:
            log.opt(colors=True).info("<green>Skipping value opt</green>, valueOptAfterNutri=FALSE")
            return self.problem
        
        # Choose full nutrition constraint or relaxed slightly to be feasible.
        if notOptimalNutrition:
            log.opt(colors=True).info("Proceeding to Value Optimization (with minimised nutri-deficiency) <yellow>Full nutrition remains infeasible!</yellow>")
        else:
            log.opt(colors=True).info("Proceeding to Value Optimization (with full nutrition constraint)")
        
        self._runValueModelTwoD(fixnutriTargetsSuccess=notOptimalNutrition)
        
        log.info(self.model.check_optimization_results())

        return self.problem


    def _runNutritionModelTwoD(self) -> None:
        cropCount = self.problem.get_crop_count()
        nutriCount = self.problem.get_nutrient_count()
        tg = self.problem.timegranularity
        
        self._buildModel(name="Nutrition_maximization")
        self._setGurobiVerbosity()
        log.start(self.model.name)
        startTime = time.time()
                
        self._addNutritionConstraints(OptimizationTarget=True)
        self._addCropTypeOverlappingContstraints()
        self._addOneDimensionalPieceWiseAreaConstraint()
                
        self.model.objective = maximize(xsum([self._nutriTargetsSuccess[p][n] for n in range(nutriCount) for p in range(tg)]) )
                                            
        self.model.write("nutri.lp")
        
        InteruptRunning = False
        status = OptimizationStatus.FEASIBLE
        smax = 10
        stepTime = time.time()
        firstTime = True
        
        # If timeoutvalue is set we are running from config files for testing. Interactive features should thus be turned of and optimize runs only once
        if self.problem.timeoutNutriAfter == 0:
            maxSeconds = 100
        else:
            maxSeconds = self.problem.timeoutNutriAfter
            # maxSecondsSame = self.problem.timeoutNutriAfter
            smax =65535
            InteruptRunning = True
        self.model.cuts=-1
        self.model.emphasis=2
        
        
        while (status == OptimizationStatus.FEASIBLE):
            status = self.model.optimize(max_solutions=smax, max_seconds=maxSeconds)
            maxSeconds=20
            smax=65535
            if time.time() - stepTime < 1:
                log.trace("sleep")
                time.sleep(0.4)
            self._setResults(status=status)
            stepTime = time.time()
                        
            periodLength = int(365/tg)
            for p in range(tg):
                log.trace("Period: {}".format(p))
                for c in range(cropCount):
                    log.trace("p:{}_c:{}_foodcropCountThisPeriod:{}".format(p,c,self._foodcropCountInPeriod[p][c].xi(0)))
                    log.trace("harvestInP:{}".format(self._isHarvestPeriod[p][c].xi(0)))
                    log.trace("StartPeriod:{}_EndOfPeriodDay:{}".format(self._isHarvestPeriod[p][c].xi(0) * (periodLength * (p)), self._isHarvestPeriod[p][c].xi(0) * (periodLength * (p+1))))
                    log.trace("")
                                
            if self.problem.status == OptimizationStatus.OPTIMAL:
                log.opt(colors=True).inf("Solution status: <green>{}</green>".format(status))
            else:
                log.opt(colors=True).inf("Solution status: <yellow>{}</yellow>".format(status))
            
            log.inf("Objective (Minimization nutrient shortage) value: {}   Optimality gap: {:.2%}".format(self.model.objective_value,self.model.gap))
            log.inf(self.problem.print_results()["selected_crop_amounts"])
            log.inf("")
            
            for c in range(cropCount):
                if self._cropCounts[c].x == self._cropCounts[c].ub:
                    log.warning("Maxed out allowed crop quantity for crop {}".format(self.problem.crops[c]))            
            
            if firstTime:
                self.problem.plot_results_2D(2)
                self.problem.plot_results_time_line(level=2)            
            else:
                if self.model.objective_value > lastObjVal:
                    self.problem.plot_results_2D(2)
            
            firstTime = False
            lastObjVal = self.model.objective_value
            if InteruptRunning and self.model.status == OptimizationStatus.FEASIBLE:
                log.warning("TIMEOUT for running nutrition optimization")
                break
            
        if sum(sum([1-self._nutriTargetsSuccess[p][n].xi(0) for n in range(nutriCount)]) for p in range(tg)) > 0:
            log.warning("Full nutrition not feasible")
            
            # Display nutritargets values: 
            log.debug("Timegranularity: {}".format(tg))
            for p in range(tg):
                log.debug("Period {}".format(p))
                log.debug("NutriTargets: {}".format([self._nutriTargetsSuccess[p][n].xi(0)  for n in range(self.problem.get_nutrient_count())] ))
                          
        time.sleep(1)
        modelRunTime = time.time() - startTime
        log.success("Runtime model {}: {}".format(self.model.name,format_timespan(modelRunTime)))
        log.end("")
        
        
    def _runValueModelTwoD(self, fixnutriTargetsSuccess: bool = False) -> None:
        self.model.clear()
        name = "Value_maximization (with minimised nutri-deficiency)" if fixnutriTargetsSuccess else "Value_maximization"
        self._buildModel(name=name)
        log.start(self.model.name)
        startTime = time.time()
        cropCount = self.problem.get_crop_count()
        nutriCount = self.problem.get_nutrient_count()
        tg = self.problem.timegranularity
        
        self._setGurobiVerbosity()
        
        self._addNutritionConstraints(OptimizationTarget=fixnutriTargetsSuccess)
        self._addCropTypeOverlappingContstraints()
        self._addOneDimensionalPieceWiseAreaConstraint()
        
        # Fix nutritargets to minimum nutrition result from nutri optimisation
        if fixnutriTargetsSuccess:
            for p in range(tg):
                for n in range(nutriCount):
                    self.model += self._nutriTargetsSuccess[p][n] >= self.problem.nutriTargetsSuccess[p][n]
        
        self._loadPreviousSolution()
        
        
        # Define the objective function as the maximisation of market value of all crops
        self.model.objective = maximize(
            xsum(self.problem.get_market_value(i) * (self._cropCounts[i] - self._cropCountsReservedForFood[i]) for i in range(cropCount)) 
        )
        self.model.write("value.lp")
                
        self.model.cuts=-1
        status = OptimizationStatus.FEASIBLE
        
        smax = 1
        
        stepTime = time.time()
        firstTime = True
        maxSeconds = 60
        maxSecondsAfter = 20
        InteruptRunning = False
        
        
        # If timeoutvalue is set we are running from config files for testing. Interactive features should thus be turned of and optimize runs only once
        if self.problem.timeoutValueAfter == 0:
            maxSeconds = 100
        else:
            maxSeconds = self.problem.timeoutValueAfter
            maxSecondsAfter = self.problem.timeoutValueAfter
            smax = 65535
            InteruptRunning = True
            self.model.cuts=-1
            self.model.emphasis=2
        
        
        while (status == OptimizationStatus.FEASIBLE):
            
            status = self.model.optimize(max_solutions=smax, max_seconds=maxSeconds)
            maxSeconds=maxSecondsAfter
            smax=65535
            self.model.emphasis=2
            self.model.cuts=3
            
            if time.time() - stepTime < 1:
                log.trace("sleep")
                time.sleep(0.4)
            self._setResults(status=status)
            stepTime = time.time()
            
            if self.problem.status == OptimizationStatus.OPTIMAL:
                log.opt(colors=True).info("Solution status: <green>{}</green>".format(status))
            else:
                log.opt(colors=True).info("Solution status: <yellow>{}</yellow>".format(status))
            
            log.inf("Objective (market) value: {}   Optimality gap: {:.2%}".format(self.model.objective_value,self.model.gap))
            log.inf(self.problem.print_results()["selected_crop_amounts"])
            
            for c in range(self.problem.get_crop_count()):
                if self._cropCounts[c].x == self._cropCounts[c].ub:
                    log.warning("Maxed out allowed crop quantity for crop {}".format(self.problem.crops[c]))            
            
            if firstTime:
                self.problem.plot_results_2D(1)
            else:
                if self.model.objective_value > lastObjVal:
                    self.problem.plot_results_2D(2)
            
            firstTime = False
            lastObjVal = self.model.objective_value
            if InteruptRunning and status == OptimizationStatus.FEASIBLE:
                log.warning("TIMEOUT for running value optimization")
                break
        
        
        # Wait before proceeding
        time.sleep(1)
        modelRunTime = time.time() - startTime
        log.success("Runtime model {}: {}".format(self.model.name,format_timespan(modelRunTime)))
        log.end("")
        
    def _loadPreviousSolution(self) -> None:
        cropCount = self.problem.get_crop_count()
        nutriCount = self.problem.get_nutrient_count()
        
        log.inf("Loading previous solution")
        self.model.start = [(self.model.var_by_name("crp_counts_{}".format(i)) , self.problem.result[self.problem.crops[i]]) for i in range(cropCount)]
        self.model.start.extend([(self.model.var_by_name("crp_counts_nutri_{}".format(i)) , self.problem.result[self.problem.crops[i]]) for i in range(cropCount)])
        self.model.start.extend([(self.model.var_by_name("nutritargets_{}_{}".format(p,n)) , self.problem.nutriTargetsSuccess[p][n]) for n in range(nutriCount) for p in range(self.problem.timegranularity)])
        
        log.inf("Validating previous solution")
        self.model.validate_mip_start()
        log.inf("Validating complete")
        
    
    