from enum import IntEnum
import math
from typing import List, Tuple
import configparser
from loguru import logger as log
from PIL import Image, ImageDraw, ImageColor, ImageFont
from mip import OptimizationStatus
import plotly.express as px
import pandas as pd
from pathlib import Path

class TimeGranularity(IntEnum):
    YEAR = 1
    HALFYEAR = 2
    MONTHS4 = 3
    QUARTER = 4

class ProblemDef():
    """Problem definition of garden, nutrients and crops. Used to query input for use in MILP solver"""
    name: str
    crops: List[str]
    maxTimesCropUsed: int
    nutrients: List[str]
    resources: List[str]
    config: configparser.ConfigParser
    result: dict
    reservedForConsumption: dict
    positions: List[Tuple[int,int]]
    cropStartDay: List[int]
    cropEndDay: List[int]
    cropTypeBorders: list[list[int]]
    nutriTargetsSuccess: list[list[float]]
    plotCount: int
    status: OptimizationStatus = None
    spaceUsage: dict
    cropStartDay: List[int]
    yields: dict
    marketValues: dict
    nutritionRequiredPerPeriod: dict
    nutritionPerPeriod: dict
    nutritionDeficitPerPeriod: dict
    daysToFarm: int
    colours: List[List[int]]
    plottingLevel: int
    timegranularity: TimeGranularity = TimeGranularity.YEAR
    valueOptAfterNutriFailure: bool
    valueOptAfterNutri: bool
    resultConclusion : dict

    def __init__(self, configFile=None) -> None:
        self.config = configparser.ConfigParser(strict=True)
        self.result = dict()
        self.resultConclusion = {"Nutrition_maximization":"undefined","Value_maximization (with minimised nutri-deficiency)":"undefined", "Nutrition_minspace":"undefined"}
        self.reservedForConsumption = []
        self.plotCount = 0
        self.daysToFarm = 365
        self.plottingLevel=1
        
        if(configFile is None):
            self.config.read('config/units.ini')
            self.config.read('config/crops.ini')
            self.config.read('config/requirements.ini')
        else:
            self.config.read(configFile)
            
        # Set name and solver
        self.name = self.config['Value Selection']['name']
        self.solver = self.config['Solver Parameters']['solver']
        self.valueOptAfterNutri = self.config['Solver Parameters'].getboolean('valueOptAfterNutri')
        self.valueOptAfterNutriFailure = self.config['Solver Parameters'].getboolean('valueOptAfterNutriFailure')
        
        # Set crops and nutritions that are going to be used in calculations
        self.crops = sorted(self.config['Value Selection']['Crops'].split(','), key=str.lower)
        self.nutrients = sorted(self.config['Value Selection']['Nutrients'].split(','), key=str.lower)
        
        # Replicate crops in config file to allow for replanting of crops
        self.maxTimesCropUsed = self.config.getint('Value Selection','maxtimescropused',fallback=1)
        self.timeoutValueAfter = self.config.getint('Solver Parameters','TimeoutValueAfter',fallback=0)
        self.timeoutNutriAfter = self.config.getint('Solver Parameters','TimeoutNutriAfter',fallback=0)
        origLength = len(self.crops)
        for repeat in range(self.maxTimesCropUsed-1):
            copyCrops = [self.crops[i]+str(repeat+2) for i in range(origLength)]
            
            for i in range(origLength):
                self.config[copyCrops[i]] = self.config[self.crops[i]]
                
            self.crops = self.crops + copyCrops
        
        self.config.set('Value Selection','Crops',','.join(self.crops))
        self.timegranularity = TimeGranularity(int(self.config['Value Selection']['TimeGranularity']))
        
        # Set resource availability like Area and Seed cost
        self.resources = sorted(self.config['Resources']['Resources'].split(','), key=str.lower)
        
        # Get colours for plotting
        if str(self.config['Drawing']['ColorType']) == 'hex':
            self.colours = [ImageColor.getcolor('#' + c,"RGB") for c in self.config.get('Drawing','Colours').split(',')]
        elif str(self.config['Drawing']['ColorType']) == 'rgb':
            self.colours = [(int(s[0]),int(s[1]),int(s[2])) for s in [c.split() for c in self.config.get('Drawing','Colours').split(',')]]
        else:
            log.error("Colour input type was not 'hex' or 'rgb'! No colors read!")


    def get_crop_count(self) -> int:
        return len(self.crops)
    def get_nutrient_count(self) -> int:
        return len(self.nutrients)
    def get_resource_count(self) -> int:
        return len(self.resources)
    def get_garden_width(self) -> int:
        return float(self.config['Resources']['GardenWidth']) *100
    def get_garden_height(self) -> int:
        return float(self.config['Resources']['GardenHeight']) *100
    
    def get_crop_plant_earlyest_day(self, cropIndex) -> int:
        return int(self.config[self.crops[cropIndex]]['PlantEarlyest'])
    def get_crop_harvest_latest_day(self, cropIndex) -> int:
        return int(self.config[self.crops[cropIndex]]['HarvestLatest'])
    def get_crop_days_to_maturatity(self, cropIndex) -> int:
        return int(self.config[self.crops[cropIndex]]['DaysToMaturity'])
    
    def get_market_value(self, cropIndex: int) -> float:
        crop = self.crops[cropIndex]
        return float(self.config[self.crops[cropIndex]]['MarketValue']) * self.__convertMassToGrams(self.config[crop]['Yield'])
    
    def get_crop_resource_requirement(self, cropIndex: int, resourceIndex: int) -> float:
        """Quantity of a given resource that is required to grow a given crop"""
        value = float(self.config[self.crops[cropIndex]][self.resources[resourceIndex]]) *10000
        return value
        
    def get_crop_nutrition(self, cropIndex: int, nutritientIndex: int) -> float:
        """Quantity of a given nutrients in grams contained in harvest of 1 plant of given crop type"""
        nutritionper100grams = self.__convertMassToGrams(self.config[self.crops[cropIndex]][self.nutrients[nutritientIndex]])
        yieldPerPlant = self.__convertMassToGrams(self.config[self.crops[cropIndex]]['Yield'])
        return (yieldPerPlant * (nutritionper100grams /100 ))
    
    def get_nutrient_priority(self, nutritientIndex: int) -> float:
        return self.config["Daily nutriprio"][self.nutrients[nutritientIndex]]
    
    def get_crop_anual_nutrient_minimum(self, nutrientIndex: int) -> float:
        """Quantity of a given nutrient that must be produced annually""" 
        return self.daysToFarm * self.__convertMassToGrams(self.config['Daily nutrimin'][self.nutrients[nutrientIndex]])
    
    
    def get_total_space_used(self) -> str:
        """Get total garden area used in m2"""
        return str(sum([float(v.split(' ')[0]) for v in self.spaceUsage.values()])) + " m2"
        
    def set_results(self, status: OptimizationStatus, result: list[int], positions: list[list[(int,int)]], cropStartDay: list[int], cropEndDay: list[int], cropTypeBorders: list[list[float]], reservedForConsumption:list[int], nutritionPerPeriod:list[list[float]], nutriTargetsSuccess: list[list[float]], name: str) -> None:
        self.name = name
        self.status=status
        self.result = dict()
        self.resultConclusion[self.name] = status.name
        self.reservedForConsumption = dict()
        self.spaceUsage = dict()
        self.yields = dict()
        self.marketValues = dict()
        self.nutritionPerPeriod = dict() 
        self.nutritionRequiredPerPeriod = dict()
        self.positions = positions
        self.cropStartDay = cropStartDay
        self.cropEndDay = cropEndDay
        self.cropTypeBorders = cropTypeBorders
        self.nutriTargetsSuccess = nutriTargetsSuccess
        self.nutritionDeficitPerPeriod = dict()
        
        for i in range(len(result)):
            key=self.crops[i]
            value=result[i]
            
            self.result.update({key:value})
            self.spaceUsage.update({key:str(float(self.config[key]['Area']) * value) + ' m2'})
            self.yields.update({key:str(self.__convertMassToGrams(self.config[key]['Yield']) * value) + ' g'})
            
            relPartForSale = 0 if value == 0 else (value - reservedForConsumption[i]) / value
            self.marketValues.update({key:(self.__convertMassToGrams(self.yields[key]) * relPartForSale) * float(self.config[key]['MarketValue'])})
            self.reservedForConsumption.update({key:reservedForConsumption[i]})
        
        for n in range(self.get_nutrient_count()):
            self.nutritionRequiredPerPeriod.update({self.nutrients[n]:self.get_crop_anual_nutrient_minimum(n) / self.timegranularity})
        
        for p in range(len(nutritionPerPeriod)):
            self.nutritionPerPeriod.update({p:{}})
            self.nutritionDeficitPerPeriod.update({p:{}})
            for n in range(self.get_nutrient_count()):
                self.nutritionPerPeriod[p].update({self.nutrients[n]:nutritionPerPeriod[p][n]})
                self.nutritionDeficitPerPeriod[p].update({self.nutrients[n]:self.nutritionRequiredPerPeriod[self.nutrients[n]] - nutritionPerPeriod[p][n]})

    def print_results(self) -> dict:
        if self.status is None:
            log.error("No solution found! Result is empty!")
            return { "resultConclusion":self.resultConclusion }
        return {
            "name":self.name,
            "resultConclusion":self.resultConclusion,
            "total_crops":sum([int(v) for v in self.result.values()]),
            "total_space":self.get_total_space_used(),
            "selected_crop_amounts":self.result,
            "crop_crow_time_range":list(zip(self.cropStartDay,self.cropEndDay)),
            "space_by_crop_type":self.spaceUsage,
            "yield_by_crop_type":self.yields,
            "value_by_crop_type":self.marketValues,
            "nutrition_required_per_period":self.nutritionRequiredPerPeriod,
            "nutrition_produced_per_period":self.nutritionPerPeriod,
            "nutrition_deficit_per_period":self.nutritionDeficitPerPeriod,
        }
        
       
    def plot_results_time_line(self, level: int=2, saveLocation: str = None) -> None:        
        if self.status is None:
            log.debug("No solution, avoiding plotting")
            return       
        
        crpcnt = self.get_crop_count()
        usedCropIndices = [c for c in range(crpcnt) if self.result[self.crops[c]] > 0 ]
        uniqueCropTypes = crpcnt/self.maxTimesCropUsed        
        uniqueCropNames = [self.crops[c] if c < uniqueCropTypes else self.crops[c][:-1] for c in range(crpcnt)]
        
        df = pd.DataFrame([
            dict(Crop=uniqueCropNames[c], Start=self.cropStartDay[c], Finish=self.cropEndDay[c], color=uniqueCropNames[c]) for c in usedCropIndices #color=uniqueCropNames[c]
        ])
        df['Day'] = df['Finish'] - df['Start'] # Needed to display day nr instead of dates
        
        # Prepare colors for plot
        colors = []
        for c in usedCropIndices:
            cindex = c%int((len(self.crops))/self.maxTimesCropUsed)
            cl = (self.colours[cindex][0],self.colours[cindex][1],self.colours[cindex][2])
            colors.append('rgb({},{},{})'.format(cl[0],cl[1],cl[2]))
            
        fig = px.bar(df, base = "Start", x = "Day", y = "Crop", orientation = 'h', text="Day")
        fig.update_yaxes(autorange="reversed") 
        
        changeDays = set([self.cropStartDay[c] for c in usedCropIndices] + [self.cropEndDay[c] for c in usedCropIndices])
        
        seen = []
        for c in changeDays:
            if c not in seen:
                fig.add_vline(x=c,line_width=1, line_dash="dash", annotation_text=str(int(c))) #annotation=self.cropStartDay[c]
                seen.append(c)
                
        fig.update_traces(marker_color=colors)
        fig.update_traces(textposition="inside",insidetextanchor="middle")
        
        
        if saveLocation:
            fig.write_image(str(saveLocation) + "/" + self.name + "-timeline.svg")
            fig.write_html(str(saveLocation) + "/" + self.name + "-timeline.html")
            # fig.write_json(str(saveLocation) + "/" + self.name + "-timeline.json")
            
        # Only show if allowed by settings
        if self.plottingLevel < level:
            return
        fig.show()
        
        
    def plot_results_2D(self, level: int=1, saveLocation: str = None) -> None:
        if self.status is None:
            log.debug("No solution, avoiding plotting")
            return       
        
        log.trace("plot? {} drempel {} arg {}".format(self.plottingLevel>=level,self.plottingLevel,level))
        wh = (int(self.get_garden_width()), int(self.get_garden_height()))
        cropWidths = [math.sqrt(self.get_crop_resource_requirement(i,0)) for i in range(self.get_crop_count())]
        shapes = [
                [
                    [
                        ProblemDef.__toImageSize((self.positions[u][i][0], self.positions[u][i][1])),
                        ProblemDef.__toImageSize((self.positions[u][i][0] + cropWidths[u], self.positions[u][i][1] + cropWidths[u]))
                    ]
                    for i in range(len(self.positions[u]))
                ]
                for u in range(self.get_crop_count())
            ]
            
        eventDays = list(filter(lambda x: x > 0 and x!=-1, sorted(set(self.cropStartDay + self.cropEndDay))))
                
        # creating new Image object
        whScaled = ProblemDef.__toImageSize(tup=wh)
        screenwh = ((whScaled[0]+50) * len(eventDays) +500,wh[1]+1750)
        img = Image.new("RGBA", screenwh)
        img1 = ImageDraw.Draw(img, mode="RGBA")  
        
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 70)
        fntA = ImageFont.truetype("Pillow/Tests/fonts/OpenSans-Regular.ttf", 80)
        
        moneyFont = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
        
        if 1 not in eventDays:
            eventDays = [1] + eventDays
        if 365 not in eventDays:
            eventDays = eventDays + [365]
        
        xoffset = 0
        gardenImageSize = ProblemDef.__toImageSize(wh)
        lastGardenNRPeriodChanged = 0
        period = 0
        periodLength = 365 / self.timegranularity
        for GardenChangeNR in range(len(eventDays)):
            day = eventDays[GardenChangeNR]
            log.trace("Draw day {}".format(day))
            
            # Draw garden space
            log.trace("Drawing garden space")
            gardenImageSize = ProblemDef.__toImageSize(wh)
            gardenImageSize = (gardenImageSize[0] + xoffset, gardenImageSize[1])

            img1.rectangle([(2+xoffset,2), gardenImageSize], fill=(5,20,5,200), outline ="green",width=2)   # Draw garden rectangles
            
            # Draw day number
            blaC = ImageColor.getrgb("black")
            whiT = ImageColor.getrgb("white")
            img1.text(xy=(5 + (whScaled[0]+10) * GardenChangeNR,whScaled[1]-20),fill=(215,215,215), stroke_fill=(0, 0, 0), stroke_width=5, font=fntA, text=str(int(day)))
            
            # Draw timeline 
            if day / periodLength  >= period+1 or GardenChangeNR == len(eventDays)-1 or GardenChangeNR == len(eventDays)-1 or (eventDays[GardenChangeNR+1] == 365 and period+1<self.timegranularity):
                
                xycord = ((12+ (lastGardenNRPeriodChanged) * (10+ProblemDef.__toImageSize(wh)[0]),whScaled[1]+80),((whScaled[0]+10) * (GardenChangeNR + (GardenChangeNR == len(eventDays)-1)) -10,whScaled[1]+150))
                
                fntPeriod = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 85)
                img1.rounded_rectangle(xy=xycord, fill=(100,100,100,245), outline ="black", width=3, radius=9)     # Draw time period 
                img1.text(xy=(xycord[0][0] +5,xycord[0][1]+35), fill=blaC, outline=(blaC),font=fntPeriod,text="Period " + str(period+1), anchor="lm", stroke_fill=(0, 0, 0), stroke_width=3)
                lastGardenNRPeriodChanged = GardenChangeNR
                period+=1
                
            for u in range(len(shapes)):
                
                c = self.colours[u%int((len(self.crops))/self.maxTimesCropUsed)]
                
                mutedC = tuple([int(0.5*c[i]) for i in range(3)])
                                
                # Draw legend text
                log.trace("Drawing legend")
                if xoffset == 0 and u < len(self.crops)/self.maxTimesCropUsed:
                    img1.text(xy=(2,whScaled[1]+220+u*50),fill=c, font=fnt, stroke_fill=mutedC, stroke_width=6, text=str(list(self.result)[u]))
                
                if self.__cropCurrentlyActive(day, u):
                    log.trace("Drawing active crop {}".format(u))
                    
                    # Draw borders of croptype
                    if list(self.result.values())[u] > 0:
                        mins = ProblemDef.__toImageSize((xoffset + self.cropTypeBorders[0][u], self.cropTypeBorders[1][u]))
                        maxs = ProblemDef.__toImageSize((xoffset + self.cropTypeBorders[2][u], self.cropTypeBorders[3][u]))
                        
                        log.trace("xminymin: {} xmaxymax: {}".format(mins,maxs))
                        maxs= (maxs[0] + 1, maxs[1]+ 1)
                        
                        img1.rectangle((mins,maxs), outline=c, width=1)
                    # Draw crops
                    for i in range(len(shapes[u])):
                        xyCord = [(shapes[u][i][0][0] + xoffset, shapes[u][i][0][1]), (shapes[u][i][1][0] + xoffset, shapes[u][i][1][1])]
                        
                        if i < self.reservedForConsumption[self.crops[u]]:    
                            img1.ellipse(xyCord, fill=(int(0.9*c[0]),int(0.9*c[1]),int(0.9*c[2]),255), outline=(c))
                        else:
                            try: # Bug in rounded rectangle related to rounded corners may occur.
                                img1.rounded_rectangle(xy=xyCord, fill=mutedC, outline=(c), radius=9)
                                # '¤' Currency sign (In Unicode	U+00A4) to indiciate unspecified currency (https://en.wikipedia.org/wiki/Currency_sign_(typography))
                                img1.text(xy=xyCord[0], fill=c, outline=(c),font=moneyFont,text='¤')
                            except: # Use normal rectangle instead
                                log.error("Wrong Shape?: {}".format(xyCord))
                                log.error("RGB values used: {} {} {}".format(mutedC[0],mutedC[1],mutedC[2]))
                                log.error("img1.rounded_rectangle({}, fill=({},{},{},255), outline=(\"{}\"), radius=7)".format(shapes[u][i],mutedC[0],mutedC[1],mutedC[2],c))
                                img1.rectangle(xyCord, fill=(mutedC[0],mutedC[1],mutedC[2],255), outline=(c))
                    
            xoffset = gardenImageSize[0]+10
        
        # Do a lot of things again so cropnames get written last on top of gardens.
        xoffset = 0
        gardenImageSize = ProblemDef.__toImageSize(wh)
        lastGardenNRPeriodChanged = 0
        for GardenChangeNR in range(len(eventDays)):
            day = eventDays[GardenChangeNR]
            gardenImageSize = ProblemDef.__toImageSize(wh)
            gardenImageSize = (gardenImageSize[0] + xoffset, gardenImageSize[1])
            
            for u in range(len(shapes)):   
                c = self.colours[u%int((len(self.crops))/self.maxTimesCropUsed)]
                mutedC = tuple([int(0.3*c[i]) for i in range(3)])
                   
                if self.__cropCurrentlyActive(day, u):
                    log.trace("Drawing active crop {}".format(u))
                    # Draw borders of croptype
                    if list(self.result.values())[u] > 0:
                        mins = ProblemDef.__toImageSize((xoffset + self.cropTypeBorders[0][u], self.cropTypeBorders[1][u]))
                        maxs = ProblemDef.__toImageSize((xoffset + self.cropTypeBorders[2][u], self.cropTypeBorders[3][u]))
                        maxs= (maxs[0] + 1, maxs[1]+ 1)
                    
                    number = str(1+int(u/(len(self.crops)/self.maxTimesCropUsed)))
                    xyCordNumber = [(maxs[0] + mins[0])/2, (maxs[1]+mins[1])/2]
                    sizeScaler = min((maxs[0] - mins[0]) * 0.7, maxs[1] - mins[1])      

                    swNum = 3
                    swText = int(sizeScaler/20)
                    fntNum = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 12 + (maxs[0] - mins[0] + maxs[1] - mins[1])/10)
                    fntB = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30 + sizeScaler )
                    anchText = "mm"
                    anchNum = "lt"
                    
                    if sizeScaler <= 50:
                        swText = 6
                    
                    if maxs[0] - mins[0] <= 50:
                        fntB = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 70)
                        fntNum = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 50)
                        anchText = "lt"
                        anchNum = "lt"
                    
                    img1.text(xy=xyCordNumber, fill=c, outline=(whiT),font=fntB,text=self.crops[u][0]+self.crops[u][1], anchor=anchText, stroke_fill=(0, 0, 0), stroke_width=swText)
                    img1.text(xy=(mins[0],mins[1]), fill=whiT, outline=(whiT),font=fntNum,text=number, anchor=anchNum, stroke_fill=(0, 0, 0), stroke_width=swNum)
            xoffset = gardenImageSize[0] + 10
            
                
        self.plotCount+=1
        
        if saveLocation:
            fp = str(saveLocation) + "/" + self.name + ".PNG"
            img.save(fp)
        
        # Only show if allowed by settings
        if self.plottingLevel < level:
            return
        img.show()
       


    def write_config(self, path: str) -> None:
        folder = Path(path)
        if not folder.exists():
            folder.mkdir(parents=True)
        # log.error(path + '/configfile.ini')
        with open(path + '/configfile.ini', 'w') as configfile:
            self.config.write(configfile)
    
    
    def write_testing_configs(self) -> None:
        """ 
        Write testing scenarios to './TestScenarios'.
        
        Nutritional tests are written to './TestScenarios/Nutri/8crops-6nutri-{?X?}x{?Y?}/{?P?}' for each x,y (garden sizes), and p (different timegranularities)
        
        Value after nutri tests are written to ./TestScenarios/Value/...
        
        """
        nutriScenarios = "TestScenarios/Nutri/8crops-8nutri-{}x{}"
        valueScenarios = "TestScenarios/Value/8crops-8nutri-{}x{}"
        
        log.inf("Nutritional test scenarios")
        # Default nutri-test crops, nutrients and crop repeats
        self.config.set('Value Selection','crops',"Eggplant,Lettuce,Okra,Potatoes,SweetPotatoes,Tomatoes,Zucchini,RedBellPepper")
        self.config.set('Value Selection','maxtimescropused',"2")
        self.config.set('Value Selection','TimeGranularity',"1")
        self.config.set('Value Selection','nutrients',"Carbohydrates,Protein,Iron,A,B1,C,Zinc,E")
        
        self.config.set('Solver Parameters','valueOptAfterNutri',"False")
        self.config.set('Solver Parameters','valueOptAfterNutriFailure',"False")
        self.config.set('Solver Parameters','TimeoutNutriAfter',"600")
        self.config.set('Solver Parameters','TimeoutValueAfter',"600")
        
        for y in range(5,8):
            for x in range(4,y+1):
                log.inf("GardenSize {}:{} with timegranularity of 1to4".format(x,y))
                s = nutriScenarios.format(x,y) 
                self.config.set('Resources','GardenWidth',str(x))
                self.config.set('Resources','GardenHeight',str(y))
            
                for p in range(4):
                    self.config.set('Value Selection','TimeGranularity',str(p+1))
                    path = s + "/" + str(p+1) + "p"
                    self.write_config(path=path)
        
        
        # Default value-test scenarios for 4 timegranularities for gardens of 5x6
        log.inf("Value test scenarios")
        self.config.set('Value Selection','crops',"Eggplant,Lettuce,Okra,Potatoes,SweetPotatoes,Tomatoes,Zucchini,RedBellPepper") # Same as Nutri-test scenarios
        self.config.set('Value Selection','maxtimescropused',"2")
        self.config.set('Value Selection','nutrients',"Carbohydrates,Protein,Iron,A,B1,C,Zinc,E") # Same as Nutri-test scenarios
        
        self.config.set('Solver Parameters','valueOptAfterNutri',"True")
        self.config.set('Solver Parameters','valueOptAfterNutriFailure',"True")
        self.config.set('Solver Parameters','TimeoutNutriAfter',"1200")
        self.config.set('Solver Parameters','TimeoutValueAfter',"1200") 
        
        x=5
        y=6
        self.config.set('Resources','GardenWidth',str(x)) 
        self.config.set('Resources','GardenHeight',str(y))
        log.inf("GardenSize {}:{} with timegranularity of 1to4".format(x,y))
        
        s = valueScenarios.format(x,y) 
        for p in range(4):
            self.config.set('Value Selection','TimeGranularity',str(object=p+1))            
            path = s + "/" + str(p+1) + "p"
            self.write_config(path=path)

    # Helper functions
    def __convertMassToGrams(self, value: str) -> float:
        """Convert values from config to grams. {Value} should be a string of the form: 'floatvalue unit'. Like this: '124.4 mg' """
        mass, unit = value.split(' ')
        return float(mass) * float(self.config['Mass Units'][unit])
    
    def __cropCurrentlyActive(self, day: int, cropIndex: int) -> bool:
        return self.cropStartDay[cropIndex] <= day and self.cropEndDay[cropIndex] > day
    
    @staticmethod
    def __toImageSize(tup: Tuple[int,int]) -> Tuple[int,int]:
        return (round(tup[0]   + 2), round(tup[1]  + 2))

    