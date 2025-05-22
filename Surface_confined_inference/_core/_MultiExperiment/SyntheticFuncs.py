import Surface_confined_inference as sci
import numpy as np
def create_times(classes, class_keys):
    for classkey in class_keys:
        times=classes[classkey]["class"].calculate_times()
        classes[classkey]["times"]=times
    return classes