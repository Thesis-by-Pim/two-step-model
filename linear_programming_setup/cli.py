"""
This is the main entrypoint to our project's code.

See the subcommands' --help options for more specific help.
"""
import os
import enum
from functools import partialmethod
import logging
import pickle
from typing import Dict, TextIO
from datetime import datetime

import click

from linear_programming_setup.mipSolver import MIPSolver
from linear_programming_setup.maxValueSolver import MaxValueSolver
from linear_programming_setup.minSpaceSolver import MinSpaceSolver
from linear_programming_setup.nutritionAndMarketValueSolver import NutritionAndMarketValueSolver
from linear_programming_setup.problemDef import ProblemDef
from loguru import logger as log
import json
from loguru_logging_intercept import setup_loguru_logging_intercept
from pathlib import Path

import sys


# default logger format: https://github.com/Delgan/loguru/issues/109
# "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

class Solvers(enum.Enum):
    NUTRITION_MIN_SPACE = MinSpaceSolver
    MAX_MARKET_VALUE = MaxValueSolver
    NUTRITION_AND_MARKET_VALUE = NutritionAndMarketValueSolver
    
class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = log.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelno, record.getMessage())
        


@click.group()
@click.option(
    '--verbosity',
    '-v',
    'level',
    type=click.Choice(['TRACE','DEBUG', 'INFO']),
    default='INFO'    
)
def cli(level) -> None:
    """
    This is the main entrypoint to our project's code.

    See the following subcommands' --help for usage.
    """
                
    logging.getLogger().disabled = True
    set_verbosity(level)
    
@cli.command("plot-instance", short_help="Read and prettyprint a garden")
@click.argument('filename',type=click.Path(exists=True))
@click.option(
    '--write',
    '-w',
    type=click.STRING,
    default=None,
    help="Write plots to location of provided path"
)
def print_instance_cmd(filename,write) -> None:
    """
    Read and plot a solution based on a solved problem instance binary
    """
    # Deserialize problemDef object binary
    with open(filename, 'rb') as f:
        problem = pickle.load(f)
    
    saveLocation=None
    # if write:
    #     saveLocation=write
    
    # Plot results (even if configuration has plotting disabled -> level = -1)
    problem.plot_results_2D(saveLocation=saveLocation,level=-1)
    problem.plot_results_time_line(saveLocation=saveLocation,level=-1)

@cli.command("generateTestScenarios")
def gen_test_configs_cmd() -> None:
    """
    Reads a garden instance and input values from a from default config files, then writes the config files for the scenarios used in testing
    """
    log.info("Parsing problem definition from config files")
    problem = ProblemDef()
    log.info("Writing testing config files to ./TestScenarios")
    problem.write_testing_configs()
    log.info("Done")
    


@cli.command("run")
@click.option(
    "--solver",
    "solver_name",
    type=click.Choice(Solvers._member_names_),  # type:ignore
    default="NUTRITION_AND_MARKET_VALUE",
    help="Available solvers are: NUTRITION_MIN_SPACE, MAX_MARKET_VALUE, NUTRITION_AND_MARKET_VALUE"
)
@click.option(
    '--plot',
    'plotting',
    type=click.IntRange(0,2),
    default=1,
    help="Choosing plotting frequency: 0: off, 1: normal, 2: plot steps of improvement"
)
@click.option(
    '--config',
    'config',
    type=click.STRING,
    default=None,
    help="Specify a config file to load. If no config file is provided, the config files in 'config/' will be loaded"
)
def run_cmd(solver_name: str, plotting: str, config: str = None) -> None:
    """
    Read offline instance, solve using MIP.

    Reads a garden instance and input values from a file, then solves the problem for
    this garden and using a MIP solver.

    Pass '-' as the filename to read from standard input instead of a file.

    """
    SaveOutput = False
    Outfolder = None
    if config:
        SaveOutput = True
        configPath = Path(config)
        if not configPath.exists():
            log.error("The given path for config file does not exist! Did you make a typo?")
            log.opt(colors=True).inf("Given path: <light-green>{}</light-green>".format(configPath))
            return
    if SaveOutput:
        Outfolder = configPath.parent / "output" / str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
        Outfolder.mkdir(parents=True)

    solver = Solvers[solver_name].value

    log.info("Parsing problem definition from config files")
    problem = ProblemDef(config)
    problem.plottingLevel = plotting
    
    
    
    

    log.info("Running solver of type {}".format(solver_name))
    instance = solver(problem=problem)
       
    resultProblem = instance.solve()
    
    outputData = resultProblem.print_results()
    print(json.dumps(outputData, indent=4))
    resultProblem.plot_results_2D(saveLocation=Outfolder)
    resultProblem.plot_results_time_line(saveLocation=Outfolder)
    
    
    
    if SaveOutput:
        f = open(Outfolder / "results.txt", mode="w")
        f.write(json.dumps(outputData, indent=4))
        f.close()
        instance.model.write(str(Outfolder) + "/solution.sol")
        instance.model.write(str(Outfolder) + "/problem.lp")
        instance.model.write(str(Outfolder) + "/problem.mps")
        # instance.model.write(str(Outfolder) + "/problem.mst")
        log.info("Model results saved to folder: " + str(Outfolder))
        # instance.model.write(str(Outfolder) + "/problem.bas")
    
        with open(Outfolder / "problemDef-obj", 'wb') as f: 
            pickle.dump(resultProblem, f)
    

def log_formatter(record):
    # print("name: "+str(record))
    # print("name: "+str(record["level"].name))
    try:
        size = os.get_terminal_size()
        line = "-" * size.columns
    except OSError as oe:
        line = "-" * 20
            
    if record["level"].name == "inf":
        return  "<green>{time:HH:mm:ss}</green> |              <level>{message}</level>\n"
    
    if record["level"].name == "START":
        return "\n<bold>" + line +"</bold>\n<green>{time:HH:mm:ss}</green> | <bold><light-green>{level: <8}</light-green></bold> | - <bold><level>RUNNING {message} model</level>\n" +  line + "</bold>\n"
    
    if record["level"].name == "END":
        return "<green>{time:HH:mm:ss}</green> | <bold><light-green>{level: <8}</light-green></bold> | <bold>\n" +  line + "</bold>\n\n\n"
    
    if record["level"].name == "INFO":
        return "\n<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | - <level>{message}</level>\n\n"
    
    return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | - <level>{message}</level>\n"

def set_verbosity(level) -> None:
    log.remove(0)
    
    # Make new level that is only used by different sink that has different formatting
    log.level("inf", no=1, color="<white>")
    log.__class__.inf = partialmethod(log.__class__.log, "inf")
    log.level(name="START", no=1, color="<white>")
    log.__class__.start = partialmethod(log.__class__.log, "START")
    log.level(name="END", no=1, color="<white>")
    log.__class__.end = partialmethod(log.__class__.log, "END")
    log.__class__.loglevel = level
    
    # Sink for normal DEBUG,INFO,ERROR etc.. logs
    log.add(sys.stderr, format=log_formatter, level="inf", filter=lambda record: record["level"].name == "inf")
    log.add(sys.stderr, format=log_formatter, level="START", filter=lambda record: record["level"].name == "START")
    log.add(sys.stderr, format=log_formatter, level="END", filter=lambda record: record["level"].name == "END")
    log.add(sys.stderr, format=log_formatter, level=level, colorize=True)

    setup_loguru_logging_intercept(
        level=logging.DEBUG,
        modules=()
    )
    setup_loguru_logging_intercept(
        level=logging.ERROR,
        modules=()
    )
    setup_loguru_logging_intercept(
        level=logging.WARNING,
        modules=()
    )
    
if __name__ == "__main__":
    cli()


