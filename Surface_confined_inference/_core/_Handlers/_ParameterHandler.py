import Surface_confined_inference as sci
class ParmeterHandler:
    def __init__(self, optim_list, boundaries, options, fixed_parameters):
        self.options=options
        self.fixed_parameters=fixed_parameters
        missing_parameters = []
        if options.problem!="inverse":
            for i in range(0, len(parameters)):
                if parameters[i] not in boundaries:
                    missing_parameters.append(parameters[i])
            if len(missing_parameters) > 0:
                    raise ValueError(
                        "Need to define boundaries for:\n %s" % ", ".join(missing_parameters)
                    )
    
                
        self.dispersion_checking(parameters)
        self.simulation_dict_construction(parameters)
        self._optim_list = parameters
     def dispersion_checking(self, optim_list):
        """

        Function called when defining `_optim_list` variable, to check for the presence of dispersed parameters.
        Args:
            optim_list (list): list of str variables
        Initialises:
            class: Dispersion, responsible for veryifying that the necessary dispersion parameters are present
        Modifies:
            Options - dispersion, sets True or False
        Raises:
            Exception: if the correct distribution parameters (for example mean and standard deviation) are not provided in `_optim_list` or `fixed_parameters`

        Scans through all parameter names (i.e. those defined in optim_list and fixed_parameters) to check for the associated dispersion parameter
        Dispersion is defined by setting parameters `{parameter}_{dispersion_flag}`. For example, if E0 has a normal distribution of values, we need `E0_mean` and `E0_std` variables
        If a `{dispersion_flag}` is found in `all_parameters` (which is the combination of `_optim_list` and `fixed_parameters`) we create a new `{variable}:list` pair in `disp_param_dict`
        This means the other `{dispersion_flag}` associated with that parameter will be added to the list associated with that `{variable}`
        Once all parameters have been scanned through, we check to see which distribution each `{variable}:list` pair is associated with, which is achieved through set comparison
        The `Dispersion` class is then initialised using the following information:
        The number of bins for each dipseresed parameter (set in `_internal_options`)
        The distribution for each parameter
        The `{variables}` with which each distribution is associated

        The required information is also initiliased for Gauss-Hermite quadratture, if required. If no normal distribution is found, then the GH_quadrature information is set to None

        """
        all_parameters = list(
            set(optim_list + list(self.fixed_parameters.keys()))
        )
        disp_flags = [
            ["mean", "std"],
            ["logmean", "logscale"],
            ["lower", "upper"],
            ["mean", "std", "skew"],
            ["logupper", "loglower"],
        ]  # Set not name must be unique
        all_disp_flags = list(set(itertools.chain(*disp_flags)))
        disp_check = [
            any(flag in param for flag in all_disp_flags) for param in all_parameters
        ]
        if any(disp_check):
            dispersion = True

            distribution_names = [
                "normal",
                "lognormal",
                "uniform",
                "skewed_normal",
                "log_uniform",
            ]
            distribution_dict = dict(
                zip(distribution_names, [set(x) for x in disp_flags])
            )
            disp_param_dict = {}
            for i in range(0, len(all_parameters)):
                for j in range(0, len(all_disp_flags)):
                    if all_disp_flags[j] in all_parameters[i]:
                        try:
                            m = re.search(
                                ".+?(?=_" + all_disp_flags[j] + ")", all_parameters[i]
                            )
                            param = m.group(0)
                            if param in disp_param_dict:
                                disp_param_dict[param].append(all_disp_flags[j])
                            else:
                                disp_param_dict[param] = [all_disp_flags[j]]
                        except:
                            continue
            dispersion_parameters = list(disp_param_dict.keys())
            dispersion_distributions = []
            distribution_keys = list(distribution_dict.keys())
            for param in dispersion_parameters:
                param_set = set(disp_param_dict[param])
                found_distribution = False
                difference = [
                    len(distribution_dict[key] - param_set) for key in distribution_keys
                ]
                min_diff = min(difference)
                best_guess = distribution_keys[difference.index(min_diff)]
                if min_diff != 0:
                    raise Exception(
                        f"Dispersed parameter {param}, assuming the distribution is {best_guess} requires {[f'{param}_{x}' for x in list(distribution_dict[best_guess])]} but only {[f'{param}_{x}' for x in list(param_set)]} found"
                    )
                else:
                    dispersion_distributions.append(best_guess)
            orig_GH_value = self.options.GH_quadrature
            if len(dispersion_distributions)!=len(self.options.dispersion_bins):
                raise ValueError("Need one bin for each of {0}, currently only have {1}".format(dispersion_distributions, self.options.dispersion_bins))
            if "normal" not in dispersion_distributions:
                self.options.GH_quadrature = False
            if self.options.GH_quadrature == True:
                self._internal_memory["GH_values"] = self.GH_setup(
                    dispersion_distributions
                )
            else:
                self._internal_memory["GH_values"] = None
            self.options.GH_quadrature = orig_GH_value
            disp_class = sci.Dispersion(
                self.options.dispersion_bins,
                dispersion_parameters,
                dispersion_distributions,
                all_parameters,
                self.fixed_parameters,
            )
            return dispersion, disp_class
        else:
            dispersion=False
            return self.options.dispersion, None
    def change_normalisation_group(self, parameters, method):
        """
        Args:
            parameters (list): list of numbers
            method (str): flag indicating if the value is to be normalised or un-normalised
        Returns:
            list: list of appropriately transformed values

        Note - this convenience method assumes the values are the parameters associated with the current parameters in optim_list
        """
        normed_params = copy.deepcopy(parameters)
        if method == "un_norm":
            for i in range(0, len(parameters)):
                normed_params[i] = sci.un_normalise(
                    normed_params[i],
                    [
                        self.boundaries[self._optim_list[i]][0],
                        self.boundaries[self._optim_list[i]][1],
                    ],
                )
        elif method == "norm":
            for i in range(0, len(parameters)):
                normed_params[i] = sci.normalise(
                    normed_params[i],
                    [
                        self.boundaries[self._optim_list[i]][0],
                        self.boundaries[self._optim_list[i]][1],
                    ],
                )
        return normed_params
