#!/usr/bin/env python
"""Provides Generic Classes to make an image analysis.

"""

from abc import ABC, abstractmethod
import pandas as pd

__author__ = "Kevin Cortacero"
__copyright__ = "Copyright 2020, Cancer Image Analysis"
__credits__ = ["Kevin Cortacero"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kevin Cortacero"
__email__ = "kevin.cortacero@inserm.fr"
__status__ = "Production"


class InputData(ABC):
    def __init__(self, data):
        self._content = data

    @abstractmethod
    def read(self):
        pass


class CohortDataFrame(InputData):
    def __init__(self, data):
        super(CohortDataFrame, self).__init__(data)

    def read(self):
        for _, row in self._content.iterrows():
            print(row)
            filepath = row.path
            name = row.id
            if row.todo == 1 and filepath != 0:
                yield (name, filepath)


class AnalysisProcedure(object):
    '''
    '''
    def __init__(self):
        self.__input_data = None
        self.__procedure = None
        self.__export_method = None
        self.__output_destination = None

    def set_input_data(self, input_data):
        self.__input_data = input_data

    def set_procedure(self, procedure):
        self.__procedure = procedure

    def set_output_destination(self, output_destination):
        self.__output_destination = output_destination

    def set_export_method(self, export_method):
        self.__export_method = export_method

    def run(self):
        print('running analysis !!')

        all_results = {}

        for (name, filepath) in self.__input_data.read():
            print(name)
            result = self.__procedure.run(filepath)
            results_df = pd.DataFrame(result, columns=result[0].keys())
            all_results[name] = results_df
            results_df.to_csv(name + '_features.csv')
            print(results_df)
        return all_results
