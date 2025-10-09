import unittest
from unittest.mock import patch
import pytest
import Surface_confined_inference as sci
import numpy as np
import os
class TestMultiExperiment:
    def setup_method(self, method):
        experiments_dict={}
        sw_freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
        zero_sw={"potential_window":[-0.425-0.15, -0.425+0.15], "thinning":10, "smoothing":20}

        dictionary_list=[
            {'E_start': np.float64(-0.8901199635261801), 'E_reverse': np.float64(-0.0006379235223668012), 'omega': np.float64(3.03471913390979), 'phase': np.float64(6.05269501468929), 'delta_E': np.float64(0.2672159297976847), 'v': np.float64(0.05953326889446427)},
            {'E_start': np.float64(-0.8900244598847762), 'E_reverse': np.float64(-0.0006520099910067856), 'omega': np.float64(9.104193706642272), 'phase': np.float64(5.682042161157082), 'delta_E': np.float64(0.22351633655321063), 'v': np.float64(0.059528212300390126)},
            {'E_start': np.float64(-0.8900404672710482), 'E_reverse': np.float64(-0.0006392975440194792), 'omega': np.float64(15.173686024700986), 'phase': np.float64(5.440366237427825), 'delta_E': np.float64(0.17876387449314424), 'v': np.float64(0.05953022016638514)},
        ]
        FT_options=dict(Fourier_fitting=True,
                        Fourier_window="hanning",
                        top_hat_width=0.25,
                        Fourier_function="abs",
                        Fourier_harmonics=list(range(3, 10)), 
                        dispersion_bins=[30],
                        optim_list=["E0","k0","gamma","Ru", "Cdl","alpha"])
        self.boundaries={
            "E0":[-0.6, -0.3],
            "k0":[1e-3, 1000],
            "gamma":[1e-11, 1e-10],
            "Ru":[1e-3, 1000],
            "Cdl":[1e-6, 5e-4],
            "CdlE1":[-1e-2, 1e-2],
            "CdlE2":[-1e-3, 1e-3],
            "CdlE3":[-1e-4, 1e-4],
            "alpha":[0.4, 0.6]
        }
        zero_ft=[-0.425, 100, 8e-11,100, 1.8e-4,0.5]
        directions=["anodic","cathodic"]
        directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
        sw_options=dict(square_wave_return="net", dispersion_bins=[30], optim_list=["E0","k0","gamma","alpha"])
        for i in range(0, len(sw_freqs)):
            
            for j in range(0, len(directions)):
                params={"omega":sw_freqs[i],
                "scan_increment":5e-3,#abs(pot[1]-pot[0]),
                "delta_E":0.8,
                "SW_amplitude":5e-3,
                "sampling_factor":200,
                "E_start":directions_dict[directions[j]]["E_start"],
                "v":directions_dict[directions[j]]["v"]}
                experiments_dict=sci.construct_experimental_dictionary(experiments_dict, {**{"Parameters":params}, **{"Options":sw_options}, "Zero_params":zero_sw}, "SWV","{0}_Hz".format(sw_freqs[i]), directions[j])
        labels=["3_Hz", "9_Hz", "15_Hz"]
        self.common={
            "Temp":278, 
            "N_elec":1,
            "area":0.036,
            "Surface_coverage":1e-10

        }
        for i in range(0, len(labels)):
            experiments_dict=sci.construct_experimental_dictionary(experiments_dict, {**{"Parameters":dictionary_list[i]}, **{"Options":FT_options}, "Zero_params":zero_ft}, "FTACV", labels[i], "280_mV")
        self.experiments_dict=experiments_dict
        self.group_list=[
           {"experiment":"FTACV",  "type":"ts", "numeric":{"Hz":{"lesser":10}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}},
           {"experiment":"FTACV", "type":"ft", "numeric":{"Hz":{"lesser":10}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}}, 
            {"experiment":"FTACV", "type":"ts", "numeric":{"Hz":{"equals":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}},
            {"experiment":"FTACV", "type":"ft", "numeric":{"Hz":{"equals":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}},
             {"experiment":"SWV", "match":["anodic"], "type":"ts", "scaling":{"divide":["omega"]}}, 
            {"experiment":"SWV", "match":["cathodic"], "type":"ts", "scaling":{"divide":["omega"]}}, 
        ]
        cwd=os.getcwd()
        if "Surface_confined_inference/tests" in cwd:
            loc=os.path.join("testdata", "multi")
            files=os.listdir(loc)
        else:
            loc=os.path.join("tests","testdata", "multi")
            files=os.listdir(loc)
        self.files=[os.path.join(loc, x) for x in files]
    def test_initialisation(self):
        cls=sci.MultiExperiment(self.experiments_dict, common=self.common, synthetic=True, normalise=True, boundaries=self.boundaries)
        assert len(cls._all_harmonics)==7
        assert cls.normalise==True
        assert cls.boundaries["k0"][0]==1e-3
        assert len(cls.class_keys)==31
    def test_grouping(self):
        cls=sci.MultiExperiment(self.experiments_dict, common=self.common, synthetic=True, normalise=True, boundaries=self.boundaries)
        cls.group_list=self.group_list
        keys=[
            "experiment:FTACV-type:ts-lesser:10Hz-equals:280mV", 
            "experiment:FTACV-type:ft-lesser:10Hz-equals:280mV", 
            "experiment:FTACV-type:ts-equals:15Hz-equals:280mV",
            "experiment:FTACV-type:ft-equals:15Hz-equals:280mV",
            "experiment:SWV-type:ts-match:anodic",
            "experiment:SWV-type:ts-match:cathodic"]
        lengths=[2, 2, 1, 1, 14, 14]
        lendict=dict(zip(keys, lengths))
        for key in keys:
            assert key in cls._grouping_keys
            assert len(cls.group_to_class[key])==lendict[key]
    def test_file_loading(self,):
        cls=sci.MultiExperiment(self.experiments_dict, common=self.common, synthetic=True, normalise=True, boundaries=self.boundaries)
        cls.group_list=self.group_list
        cls.file_list=self.files    
    def test_evaluation(self):
        cls=sci.MultiExperiment(self.experiments_dict, common=self.common, synthetic=True, normalise=True, boundaries=self.boundaries)
        cls.group_list=self.group_list
        cls.file_list=self.files  
        sim_dict=dict(zip(["E0","k0","gamma","Ru", "Cdl","alpha"],  [-0.45, 35, 8e-11, 500,1.25e-4, 0.55]))
        sim_params=[sci._utils.normalise(sim_dict[x], self.boundaries[x]) for x in cls._all_parameters]
        simulations=cls.evaluate(sim_params)
        for ckey in cls.class_keys:
            assert sci._utils.RMSE(cls.classes[ckey]["data"], simulations[ckey])<1e-2
    def test_saving_and_loading(self):
        cls=sci.MultiExperiment(self.experiments_dict, common=self.common, synthetic=True, normalise=True, boundaries=self.boundaries)
        cls.group_list=self.group_list
        cls.file_list=self.files  
        cls.save_class(include_data=True)
        loaded_class=sci.BaseMultiExperiment.from_directory("saved")
        assert loaded_class.normalise==True
        sim_dict=dict(zip(["E0","k0","gamma","Ru", "Cdl","alpha"],  [-0.45, 35, 8e-11, 500,1.25e-4, 0.55]))
        sim_params=[sci._utils.normalise(sim_dict[x], self.boundaries[x]) for x in loaded_class._all_parameters]
        simulations=loaded_class.evaluate(sim_params)
        for ckey in loaded_class.class_keys:
            assert sci._utils.RMSE(loaded_class.classes[ckey]["data"], simulations[ckey])<1e-2
if __name__ == '__main__':    
    unittest.main()
    #v=TestMultiExperiment()
    #v.setup_method(None)
    #v.test_evaluation()
    #v.test_grouping()