# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import pandas as pd
from absl.testing import absltest

from concordia.deliberate_integration.analysis.analysis_comprehensive_metrics import calculate_gini, compute_survey_kl

class AnalysisMetricsTest(unittest.TestCase):

  def test_calculate_gini_equal(self):
    """Tests that Gini coefficient of equal elements is 0."""
    array = [10, 10, 10, 10]
    # Gini should be 0 for equal distribution
    self.assertAlmostEqual(calculate_gini(array), 0.0, places=5)

  def test_calculate_gini_unequal(self):
    """Tests that Gini coefficient of unequal elements is correct."""
    array = [0, 0, 0, 100]
    # For n=4, values [0,0,0,100], Gini should be 0.75
    self.assertAlmostEqual(calculate_gini(array), 0.75, places=5)

  def test_compute_survey_kl(self):
    """Tests that Survey KL divergence is computed correctly."""
    # Create mock data
    data = {
        'Model': ['Human', 'Human', 'Model1', 'Model1'],
        'question_id': ['q1', 'q1', 'q1', 'q1'],
        'response': ['A', 'A', 'A', 'B']
    }
    df = pd.DataFrame(data)
    
    # Human: 100% 'A'
    # Model1: 50% 'A', 50% 'B'
    # KL should be positive
    
    results = compute_survey_kl(df)
    self.assertIn('Model1', results)
    self.assertGreater(results['Model1'], 0.0)

if __name__ == '__main__':
  absltest.main()
