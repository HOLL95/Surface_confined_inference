
def check_input_dict(input_dict, parameters):
    user_params=set(input_dict.keys())
    required_params=set(parameters)
    extra=user_params-required_params
    missing=required_params-user_params
    if len(missing)>0:
        print("Simulation requires the following parameters: {0}".format((" ").join(list(missing))))
    if len(extra)>0:
        print("The following parameters are not required for the simulation: {0}".format((" ").join(list(extra))))

