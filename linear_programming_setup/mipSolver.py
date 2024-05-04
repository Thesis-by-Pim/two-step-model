import math
from typing import List
from linear_programming_setup.problemDef import ProblemDef
from mip import *
from abc import ABC, abstractmethod
from loguru import logger as log




#  https://www.gurobi.com/documentation/current/refman/recommended_ranges_for_var.html



class MIPSolver(ABC):
    """
    This class models and solves a garden planting plan using MILP
    """
    model = Model   

    # Crop quantities
    _cropUsed: list[Var]
    _cropCounts: list[Var]
    _cropCountsReservedForFood: list[Var] # Part of number in _cropCounts
    _instance_lowerbounds = list[int]
    _instance_upperbounds = list[int]
    
    # 2D intervals
    _beta_y: list[list[Var]]
    _alpha_x: list[list[Var]]
    _zvars: list[list[list[Var]]]
    
    # Borders & crop areas
    _x_mins: list[Var]
    _x_maxs: list[Var]
    _y_mins: list[Var]
    _y_maxs: list[Var]
    _spaceBetweenCroptypes: float
    _spaceBetweenCrops: float
    _cropStartDay: list[Var]
    _isHarvestPeriod: list[List[Var]]
    _foodcropCountInPeriod:  list[List[Var]]
    
    # Goals
    _nutriTargetsSuccess: list[list[Var]]
    
    
    def __init__(self, problem: ProblemDef) -> None:
        self.problem = problem
        self._instance_lowerbounds = [0] * self.problem.get_crop_count()
        self._instance_upperbounds = [1000] * self.problem.get_crop_count()
        self._spaceBetweenCroptypes = 10
        self._spaceBetweenCrops = 1
        

    def solve(self) -> ProblemDef:        
        log.inf("Building model")
        self._buildModel(name=str(type(self)))
        
        return self._solve()

    # Main solver method for the ILP
    @abstractmethod
    def _solve(self) -> ProblemDef:
        pass



    def _buildModel(self, name: str) -> None:
        cropCount = self.problem.get_crop_count()            
        
        # Choose solver
        solver=self.problem.solver
        if solver == 'GRB':
            self.model = Model(name=name, solver_name=GRB)
        elif solver == 'CBC':
            self.model = Model(name=name, solver_name=CBC)
        else:
            log.error("Solver with name '{}' not found".format(solver))
        
        # Variables for the number of crops of each type
        log.trace("Instance upperbounds: {}".format(self._instance_upperbounds))
        self._cropCounts =                  [self.model.add_var(name="crp_counts_{}".format(i),var_type=INTEGER, lb=self._instance_lowerbounds[i], ub=self._instance_upperbounds[i]) for i in range(cropCount)]        
        self._cropCountsReservedForFood =   [self.model.add_var(name="crp_counts_nutri_{}".format(i),var_type=INTEGER, lb=0, ub=self._instance_upperbounds[i]) for i in range(cropCount)]
        
        for i in range(cropCount):
            self.model += self._cropCountsReservedForFood[i] <= self._cropCounts[i]
        
        # Variable for croptypes used
        self._cropUsed =  [self.model.add_var(name="crp_used{}".format(i),var_type=BINARY) for i in range(cropCount)]
        for cropType in range(cropCount):
            self.model += self._cropUsed[cropType] <= self._cropCounts[cropType]
            self.model += self._cropCounts[cropType] <= self._cropUsed[cropType] * self._instance_upperbounds[cropType] 
        
        self._cropStartDay = [
            self.model.add_var(
                name="crpstartday_{}".format(i),
                var_type=INTEGER, 
                lb=0, 
                ub=365-self.problem.get_crop_days_to_maturatity(i)
                ) 
            for i in range(cropCount)
        ]
        
        
        for c in range(cropCount):
            self.model += self._cropStartDay[c] >= self.problem.get_crop_plant_earlyest_day(i) #* self._cropUsed[i] # mult with cropused should not be required but may improve performance
            # self.model += self._cropStartDay[c] <= 365 - 365 * (1 - self._cropUsed[i]) # part seems unnececary because we use ANDvars already which include cropused
        
        # Variables for borders
        self._x_mins =  [self.model.add_var(var_type=CONTINUOUS, name="x_mins_{}".format(i), lb=0, ub=self.problem.get_garden_width()) for i in range(cropCount)]
        self._x_maxs =  [self.model.add_var(var_type=CONTINUOUS, name="x_maxs_{}".format(i), lb=0, ub=self.problem.get_garden_width()) for i in range(cropCount)]
        self._y_mins =  [self.model.add_var(var_type=CONTINUOUS, name="y_mins_{}".format(i), lb=0, ub=self.problem.get_garden_height()) for i in range(cropCount)]
        self._y_maxs =  [self.model.add_var(var_type=CONTINUOUS, name="y_maxs_{}".format(i), lb=0, ub=self.problem.get_garden_height()) for i in range(cropCount)]
                
        # Variables indicating crop area overlap
        self._cropTypeSeperationDelta = [
                [
                    [
                        self.model.add_var(name="delta_{}_{}_{}".format(cropOne,cropTwo-cropOne-1,i),var_type=BINARY) for i in range(4)
                    ] 
                    for cropTwo in range(cropOne + 1, cropCount)
                ]
                for cropOne in range(cropCount)
            ]                    
        
        self._addLinkingTimePeriodConstraints() # Initialize nutritional period-variables
    
                       
    # Operations Research Letters
    # Piecewise linear approximation of functions of two variables in MILP models
    # Claudia D'Ambrosio, Andrea Lodi, Silvano Martello
    # Other:
        # https://or.stackexchange.com/questions/180/how-to-linearize-the-product-of-two-continuous-variables
        # https://download.aimms.com/aimms/download/manuals/AIMMS3OM_IntegerProgrammingTricks.pdf
    # Improvement implemented: https://or.stackexchange.com/questions/7293/how-to-linearize-the-product-of-two-integer-variables
    #   - Reduces binary variables : 𝑀_x=⌈log2(U_x+1)⌉ for x
    
    def _addOneDimensionalPieceWiseAreaConstraint(self) -> None:
        """
        Generate a one dimensional piecewise approximation to interpolate the width of the croptype for any hight of the croptype. 
        Then interpolate between width approximation functions for each height to get the full function z = x*y
        """
        cropCount = self.problem.get_crop_count()
        gardenWidth = self.problem.get_garden_width()
        gardenHeight = self.problem.get_garden_height()
        log.error("width: {} Height: {}".format(gardenWidth,gardenHeight))
        
        if gardenWidth > gardenHeight:
            log.error("switch")
        
        self._alpha_x = [[] for i in range(cropCount)]
        self._beta_y  = []
        self._zvars = [[] for i in range(cropCount)]
    
        for cropType in range(cropCount):
            cropSize = self.problem.get_crop_resource_requirement(cropType,0)
            log.trace("CropSize: {}".format(cropSize))
            cropWidth = math.sqrt(cropSize)
            log.trace("CropWidth: {}".format(cropWidth))
            
            # ---| Setup y-axis as integer |--- #
            maxValY = int(((gardenHeight+ self._spaceBetweenCrops)/(cropWidth+ self._spaceBetweenCrops)))
            self._beta_y.append(self.model.add_var(var_type=INTEGER, lb=0, ub=maxValY, name="beta_y_{}_{}".format(cropType,0)))
            
            # Define vertical space constraints on y-var
            self.model += self._y_maxs[cropType] - self._y_mins[cropType] +self._spaceBetweenCrops * self._cropUsed[cropType] >= (cropWidth+self._spaceBetweenCrops) * self._beta_y[cropType]
            
            # |---| Setup x-axis as piecewise linear sum of each bit value of the integer that represents width |---| #
            rx = int(math.floor(math.log2(float(gardenWidth+self._spaceBetweenCrops) / (cropWidth + self._spaceBetweenCrops))+1))
            self._alpha_x[cropType] = [self.model.add_var(var_type=BINARY, name="alpha_x_{}_{}".format(cropType,r)) for r in range(rx)]
            
            log.debug("rx (bits for width var): {}".format(rx))
            
            # Define horizontal space constraints on x-var 
            self.model += self._x_maxs[cropType] - self._x_mins[cropType] +self._spaceBetweenCrops * self._cropUsed[cropType] <= (cropWidth+self._spaceBetweenCrops) * xsum([math.pow(2,i) * self._alpha_x[cropType][i]  for i in range(rx)])#+ 2*self.model.infeas_tol
            self.model += self._x_maxs[cropType] - self._x_mins[cropType] +self._spaceBetweenCrops * self._cropUsed[cropType] >= (cropWidth+self._spaceBetweenCrops) * xsum([math.pow(2,i) * self._alpha_x[cropType][i]  for i in range(rx)])#+2 * self.model.infeas_tol
            
            
            # |---| Linearize products of X𝑖Y with result zvars[i] |---| #
            self._zvars[cropType] = [self.model.add_var(var_type=INTEGER, lb=0) for _ in range(rx)]
            for i in range(rx):
                self.model += self._zvars[cropType][i] <= self._beta_y[cropType].ub * self._alpha_x[cropType][i]
                self.model += self._zvars[cropType][i] <= self._beta_y[cropType] - self._beta_y[cropType].lb  * (1 - self._alpha_x[cropType][i])
                self.model += self._zvars[cropType][i] >= self._beta_y[cropType] - self._beta_y[cropType].ub  * (1 - self._alpha_x[cropType][i])
                
            # Set cropcount equal to x*y (but with substitution by zvars)            
            self.model += self._cropCounts[cropType] <= xsum([math.pow(2,i) * self._zvars[cropType][i] for i in range(rx)]) #+2 * self.model.infeas_tol
            self.model += self._cropCounts[cropType] >= xsum([math.pow(2,i) * self._zvars[cropType][i] for i in range(rx)])
                      
                      
    # https://yetanothermathprogrammingconsultant.blogspot.com/2017/07/rectangles-no-overlap-constraints.html
    def _addCropTypeOverlappingContstraints(self):
        gardenWidth = self.problem.get_garden_width()
        gardenHeight = self.problem.get_garden_height()
        cropCount = self.problem.get_crop_count()

        # Checking if crop is being used. Needed to know if overlapping constraints should be enforced for that crop.
        for cropType in range(cropCount):
            cropWidth = math.sqrt(self.problem.get_crop_resource_requirement(cropType,0))
            self.model += self._cropCounts[cropType] <=  (self._instance_upperbounds[cropType] *  self._cropUsed[cropType] )
                    
        
        # Vars defined as BINARY AND of cropUsed[a] and cropUsed[b]
        ANDvars =           [[self.model.add_var(name="AND_{}_{}".format(i,u),var_type=BINARY)                     for u in range(i+1,cropCount)] for i in range(cropCount-1)]
        endOneMTstartTwo =  [[self.model.add_var(name="endoneMTstarttwo_{}_{}".format(i,u),var_type=BINARY, lb=0)  for u in range(i+1,cropCount)] for i in range(cropCount-1)]
        endTwoMTstartOne =  [[self.model.add_var(name="endtwoMTstartone_{}_{}".format(i,u),var_type=BINARY, lb=0)  for u in range(i+1,cropCount)] for i in range(cropCount-1)]
                       
        # Do not allow replanting of crop before it is planted a first time...
        times = self.problem.maxTimesCropUsed
        cropnumber = int(cropCount/times)

        for t in range(times-1):
            # log.debug(t)
            for c in range(cropnumber):
                c1 = c + t*cropnumber
                c2 = int(c + (t+1)*cropnumber)
                self.model += self._cropUsed[c2] <= self._cropUsed[c1]
                self.model += self._cropStartDay[c2] >= self._cropStartDay[c1] + self.problem.get_crop_days_to_maturatity(c1) * (self._cropUsed[c2])
                              
        for cropOne in range(cropCount):
            cropWidth = math.sqrt(self.problem.get_crop_resource_requirement(cropIndex=cropOne,resourceIndex=0))
            growTimeOne = self.problem.get_crop_days_to_maturatity(cropOne)
            
            self.model += self._x_maxs[cropOne] >= self._x_mins[cropOne]
            self.model += self._y_maxs[cropOne] >= self._y_mins[cropOne]
                                    
            for cropTwo in range(cropOne+1,cropCount):
                    growTimeTwo = self.problem.get_crop_days_to_maturatity(cropTwo)
                    
                    # Combining AND (AND(AB),AND(CD)) into product AND=ABCD: 3 and 4 of -> https://luuquangtrung.github.io/posts/2021/12/notes-linearization/#fn:Liberti2018
                    
                    ##### Crop overlapping in space between crop A and B is NOT allowed if(andonlyif): 
                        #   1[Both crops in use]     3AND     2[Growing periods overlap]
                        # or:
                        #   (Used_A && Used_B)      &&      ((endTimeA >= startTimeB) && (endTimeB>=startTimeA))

                    ## 1[Both crops in use]:
                    self.model += ANDvars[cropOne][cropTwo - cropOne-1] <= self._cropUsed[cropOne]
                    self.model += ANDvars[cropOne][cropTwo - cropOne-1] <= self._cropUsed[cropTwo]
                    
                    ## 2[Growing periods overlap]: 
                    # Both crops active in overlapping time periods: https://nedbatchelder.com/blog/201310/range_overlap_in_two_compares.html                    
                    self.model += ANDvars[cropOne][cropTwo - cropOne-1] <= endOneMTstartTwo[cropOne][cropTwo - cropOne-1]
                    self.model += ANDvars[cropOne][cropTwo - cropOne-1] <= endTwoMTstartOne[cropOne][cropTwo - cropOne-1]
                    self.model += ANDvars[cropOne][cropTwo - cropOne-1] >= endOneMTstartTwo[cropOne][cropTwo - cropOne-1] + endTwoMTstartOne[cropOne][cropTwo - cropOne-1] + self._cropUsed[cropOne] + self._cropUsed[cropTwo] -4 + 1
                    
                    # Define two variables to mean that overlap in time is happening if they are both TRUE
                    self.model += self._cropStartDay[cropOne] + growTimeOne >= self._cropStartDay[cropTwo] - (365)*(1-endOneMTstartTwo[cropOne][cropTwo - cropOne-1])
                    self.model += self._cropStartDay[cropTwo] + growTimeTwo >= self._cropStartDay[cropOne] - (365)*(1-endTwoMTstartOne[cropOne][cropTwo - cropOne-1])
                    # Define that overlap may not be happening if the variables are FALSE
                    self.model += self._cropStartDay[cropOne] + growTimeOne <= self._cropStartDay[cropTwo] + (365)*(endOneMTstartTwo[cropOne][cropTwo - cropOne-1])
                    self.model += self._cropStartDay[cropTwo] + growTimeTwo <= self._cropStartDay[cropOne] + (365)*(endTwoMTstartOne[cropOne][cropTwo - cropOne-1])
                    
                    ##### Crop overlapping constraints
                    self.model += ((self._x_maxs[cropOne] ) + self._spaceBetweenCroptypes <= (self._x_mins[cropTwo] + gardenWidth * self._cropTypeSeperationDelta[cropOne][cropTwo - cropOne - 1][0]))
                    self.model += ((self._x_maxs[cropTwo] ) + self._spaceBetweenCroptypes <= (self._x_mins[cropOne] + gardenWidth * self._cropTypeSeperationDelta[cropOne][cropTwo - cropOne - 1][1]))
                    self.model += ((self._y_maxs[cropOne] ) + self._spaceBetweenCroptypes <= (self._y_mins[cropTwo] + gardenHeight * self._cropTypeSeperationDelta[cropOne][cropTwo - cropOne - 1][2]))
                    self.model += ((self._y_maxs[cropTwo] ) + self._spaceBetweenCroptypes <= (self._y_mins[cropOne] + gardenHeight * self._cropTypeSeperationDelta[cropOne][cropTwo - cropOne - 1][3]))
                    
                    self.model += (xsum(self._cropTypeSeperationDelta[cropOne][cropTwo - cropOne - 1][k] for k in range(4)) <= 4 - ANDvars[cropOne][cropTwo - cropOne-1])
              


    def _addLinkingTimePeriodConstraints(self) -> None:
        """Linking nutritional periods to time of harvesting crops."""
        cropCount = self.problem.get_crop_count()
        tg = self.problem.timegranularity
        periodLength = 365/float(tg)
        
        # Adding nutrition requirement as optional so that we can optimize it by summing over _nutriTargetsSuccess
        self._nutriTargetsSuccess = [[self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="nutritargets_{}_{}".format(p,n)) for n in range(self.problem.get_nutrient_count())] for p in range(tg)]
        self._isHarvestPeriod = [[self.model.add_var(name="cropharvestoperiod_{}_{}".format(p,c),var_type=BINARY) for c in range(cropCount)] for p in range(tg)]
        self._foodcropCountInPeriod = [[self.model.add_var(name="cropsfoodperiod_{}_{}".format(p,c),var_type=INTEGER, lb=0) for c in range(cropCount)] for p in range(tg)]
        
        for c in range(cropCount):
            growTime = self.problem.get_crop_days_to_maturatity(c)
            log.trace("Crop_{}".format(c))
            for p in range(tg):
                log.trace("P_{}".format(p))

                    # Overlap happens between growing period for crop and nutritional period if both:  
                    #   -->     cropEnd >= startPeriod 
                    #   -->     endPeriod >= cropEnd
                    
                self.model += self._cropStartDay[c] + growTime                                                              >= (periodLength * p) * (self._isHarvestPeriod[p][c]) + 1
                self.model += (periodLength * (p+1)) +  (365 - (periodLength * (p+1))) * (1-self._isHarvestPeriod[p][c])    >= self._cropStartDay[c] + growTime -1                                 
                
                self.model += self._foodcropCountInPeriod[p][c] <= self._isHarvestPeriod[p][c] * self._instance_upperbounds[c]
                self.model += self._foodcropCountInPeriod[p][c] <= self._cropCountsReservedForFood[c]
                self.model += self._foodcropCountInPeriod[p][c] >= self._cropCountsReservedForFood[c] - self._instance_upperbounds[c] * (1 - self._isHarvestPeriod[p][c])
            
            self.model += xsum(self._foodcropCountInPeriod[p][c] for p in range(tg)) <= self._cropCountsReservedForFood[c]
        
        
    def _addNutritionConstraints(self, OptimizationTarget: bool = False) -> None:
        """Sets a minimum target for each nutrient as specified in the config file"""
        cropCount = self.problem.get_crop_count()
        nutriCount = self.problem.get_nutrient_count()
        tg = self.problem.timegranularity
                               
        for n in range(nutriCount):
            for c in range(cropCount):
                log.trace("Crop {} nutrient {} {} amount {}".format(self.problem.crops[c],n, self.problem.nutrients[n], self.problem.get_crop_nutrition(c,n)))
                
            for p in range(tg):
                self.model += self._nutriTargetsSuccess[p][n] >= 1 - OptimizationTarget
                self.model += self._nutriTargetsSuccess[p][n] <= 1
            
                # Enforce nutrition         
                self.model += (
                    xsum(self.problem.get_crop_nutrition(c,n) * self._foodcropCountInPeriod[p][c] for c in range(cropCount)) 
                        >= (self.problem.get_crop_anual_nutrient_minimum(n) / tg) * self._nutriTargetsSuccess[p][n] 
                )

    def _setGurobiVerbosity(self) -> None:
        self.model.verbose = 0
        if log.__class__.loglevel in ["DEBUG","TRACE"]:
            self.model.verbose = 1
            log.info("Running model with verbose output from {}".format(self.model.solver_name))
    

    def _setResults(self, status: OptimizationStatus) -> None:
        cropAmounts: list[int]
        cropAmounts = [v.xi(0) for v in self._cropCounts]
        cropAmountsNutri = [v.xi(0) for v in self._cropCountsReservedForFood]
        cropStartDay =  [-1] * self.problem.get_crop_count()
        cropEndDay =    [-1] * self.problem.get_crop_count()
        
        cropCount = self.problem.get_crop_count()
        nutriCount = self.problem.get_nutrient_count()
        
        
        for c in range(self.problem.get_crop_count()):
            if self._cropUsed[c].xi(0):
                # log.debug("cropused {}".format(self._cropUsed[c].xi(0)))
                cropStartDay[c] = self._cropStartDay[c].xi(0)
                cropEndDay[c] = cropStartDay[c] + self.problem.get_crop_days_to_maturatity(c)
                            
        # DEBUG CODE: Checking overlap in space-time is correct    
        # for c1 in range(self.problem.get_crop_count()):
        #     for c2 in range(c1,self.problem.get_crop_count()):        
        #         log.debug("AND [{},{}]: ANDvars {}    used {}    sametime {}".format(
        #             c1,
        #             c2,
        #             self.model.var_by_name(str(c1)+"AND"+str(c2)).xi(0),
        #             self.model.var_by_name(str(c1)+"ANDcropsused"+str(c2)).xi(0),
        #             self.model.var_by_name(str(c1)+"ANDsametime"+str(c2)).xi(0)
        #         ))
        # log.debug("Solutions:")
        # for u in range(self.model.num_solutions):
        #     log.debug([self._cropCounts[c].xi(u) for c in range(self.problem.get_crop_count())])
        
        if self.model.num_solutions == 0:
            log.error("No solutions found")
            return
        
        filteredPositions = []
        for c in range(self.problem.get_crop_count()):
            cropArea = self.problem.get_crop_resource_requirement(cropIndex=c,resourceIndex=0)
            cropWidth = math.sqrt(cropArea)
            filteredPositions.append([])
            
            if bool(self._cropCounts[c].xi(0) > 0):
                counter = int(self._cropCounts[c].xi(0))
                
                maxx = int((self._x_maxs[c].xi(0) - self._x_mins[c].xi(0) + self._spaceBetweenCrops)/((cropWidth*0.999+self._spaceBetweenCrops)))
                maxy = int((self._y_maxs[c].xi(0) - self._y_mins[c].xi(0) + self._spaceBetweenCrops)/((cropWidth*0.999+self._spaceBetweenCrops)))
                                
                for row in range(maxy):
                    for column in range(maxx):
                        if (counter >0):
                            filteredPositions[c].append((self._x_mins[c].xi(0) + column * (cropWidth + self._spaceBetweenCrops) , self._y_mins[c].xi(0) + row * (cropWidth+ self._spaceBetweenCrops)))
                            counter-=1
                    
        borders = [[x.xi(0) for x in self._x_mins], [x.xi(0) for x in self._y_mins], [x.xi(0) for x in self._x_maxs], [x.xi(0) for x in self._y_maxs]]
        filteredBorders = []
        for index in range(len(borders)):
            filteredBorders.append(borders[index])

        # Nutrients per produced per period
        nutritionPerPeriod = [
            [sum(self.problem.get_crop_nutrition(c,n) * self._foodcropCountInPeriod[p][c].xi(0) for c in range(cropCount)) for n in range(nutriCount)] 
            for p in range(self.problem.timegranularity)
        ]
        nutriTargetsSuccess = [[self._nutriTargetsSuccess[p][n].xi(0) for n in range(nutriCount)] for p in range(self.problem.timegranularity)]

        if(status == OptimizationStatus.OPTIMAL):
            self.problem.set_results(
                status=status, 
                result=cropAmounts, 
                positions=filteredPositions,
                cropStartDay=cropStartDay,
                cropEndDay=cropEndDay,
                cropTypeBorders=filteredBorders,
                reservedForConsumption=cropAmountsNutri,
                nutritionPerPeriod=nutritionPerPeriod,
                nutriTargetsSuccess=nutriTargetsSuccess,
                name=self.model.name
                )

        else:
            self.problem.set_results(
                status=status,
                result=cropAmounts,
                positions=filteredPositions,
                cropStartDay=cropStartDay,
                cropEndDay=cropEndDay,
                cropTypeBorders=filteredBorders,
                reservedForConsumption=cropAmountsNutri,
                nutritionPerPeriod=nutritionPerPeriod,
                nutriTargetsSuccess=nutriTargetsSuccess,
                name=self.model.name
                )
        return status