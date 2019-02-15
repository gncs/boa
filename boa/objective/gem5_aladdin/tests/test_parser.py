import os
from unittest import TestCase

import pkg_resources

from boa.objective.gem5_aladdin.process import parse_file, output_regexps, get_cycle_power_area


class TestParser(TestCase):
    RESOURCES_DIR = pkg_resources.resource_filename(__package__, 'resources')

    def test_file_does_not_exist(self):
        path = os.path.join(self.RESOURCES_DIR, 'not_a_file.yaml')
        with self.assertRaises(Exception):
            parse_file(file_path=path, regexps=output_regexps)

    def test_parse_output(self):
        path = os.path.join(self.RESOURCES_DIR, 'gem5_aladdin.output')

        expected = {
            'cycle': 51439,
            'avg_power': 39.251,
            'idle_fu_cycles': 35951,
            'avg_fu_cycles': 34.1792,
            'avg_fu_dynamic_power': 23.605,
            'avg_fu_leakage_power': 10.5742,
            'avg_mem_power': 5.07184,
            'avg_mem_dynamic_power': 3.4128,
            'avg_mem_leakage_power': 1.65904,
            'total_area': 1.37064E06,
            'fu_area': 1.04667E06,
            'mem_area': 323976,
            'num_double_precision_fp_multipliers': 37,
            'num_double_precision_fp_adders': 30,
            'num_trigonometric_units': 14,
            'num_bit-wise_operators_32': 9,
            'num_shifters_32': 6,
            'num_registers_32': 192,
        }

        results = parse_file(file_path=path, regexps=output_regexps)

        for k, v in results.items():
            self.assertAlmostEqual(v, expected[k])

    def test_cycle_power_area(self):
        path = os.path.join(self.RESOURCES_DIR, 'gem5_aladdin.output')

        expect = [51439, 39.251, 1.37064E06]
        result = get_cycle_power_area(path)

        for e, r in zip(expect, result):
            self.assertAlmostEqual(e, r)
