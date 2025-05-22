import collections.abc
import numbers
from typing import List, Optional, Sequence
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption, OptionDescriptor, DictOption,ComposedOption, FileOption, DirectoryOption
)
from ._OptionsMeta import OptionsManager, OptionsMeta
class AxInterfaceOptions(OptionsManager):
    name=StringOption("name",
                        default="AxClientOptimisation",
                        doc="Ax client optimisation name")
    log_directory=DirectoryOption("log_directory",
                    default="slurm_logs",
                    must_exist=False, 
                    must_be_empty=False,
                    doc="Location for slurm log files")
    results_directory=DirectoryOption("results_directory",
                    default="results",
                    must_exist=True,
                    must_be_empty=True,
                    can_create=True,
                    doc="Location for results of optimisation and simulation files")
    independent_runs=NumberOption("independent_runs",
                                default=10,
                                min_value=1,
                                doc="Number of independent runs"
                                )
    num_iterations=NumberOption("num_iterations",
                                default=100, 
                                min_value=10,
                                max_value=500,
                                doc="Number of iterations for ax client to run over")
    max_run_time=NumberOption("max_run_time",
                            default=48,
                            min_value=1, 
                            max_value=48,
                            doc="Amount of time in hours for run")
    num_cpu=NumberOption("num_cpu", 
                        default=1,
                        doc="Number of CPUs per submitted job")
    simulate_front=BoolOption("simulate_front",
                            default=True,
                            doc="Whether to generate simulations for all pareto front points while in slurm cluster")
    rclone_directory=StringOption("rclone_directory",
                                default="", 
                                doc="Address to attempt to rclone results directory into")
    in_cluster=BoolOption("in_cluster",
                            default=True,
                            doc="Indicates whether or not code should attempt to use submitit or not")
    email=StringOption("email",
                        default="",
                        doc="Email for slurm cluster to send updates to")
    project=StringOption("project",
                        default="",
                        doc="Project under which to cost slurm cluster submission")
    GB_ram=NumberOption("GB_ram",
                        default=8,
                        doc="Gigabytes of RAM to assign for Slurm cluster")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
