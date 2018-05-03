from h2o.estimators.xgboost import *
from tests import pyunit_utils
import unittest


class TestXGBoostMaxDepth(unittest.TestCase):
    def test_xgboost_max_depth_param(self):
        assert H2OXGBoostEstimator.available()

        prostate_frame = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate.csv'))
        x = ['RACE']
        y = 'CAPSULE'
        prostate_frame[y] = prostate_frame[y].asfactor()

        prostate_frame.split_frame(ratios=[0.75], destination_frames=['prostate_training', 'prostate_validation'], seed=1)

        training_frame = h2o.get_frame('prostate_training')

        ## MAX_DEPTH should not exceed 15.
        model = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                            booster='gbtree', seed=1, ntrees=2, distribution='bernoulli', max_depth=16)
        with self.assertRaises(Exception) as outcome:
            model.train(x=x, y=y, training_frame=training_frame)

        assert str(outcome.exception).__contains__(
                "MAX_DEPTH limit for XGBoost must be between 1 and 15. Value used: 16")

        ## MAX_DEPTH should not be below 1
        model = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7,
                                    booster='gbtree', seed=1, ntrees=2, distribution='bernoulli', max_depth=0)
        with self.assertRaises(Exception) as outcome:
            model.train(x=x, y=y, training_frame=training_frame)

        assert str(outcome.exception).__contains__(
                "MAX_DEPTH limit for XGBoost must be between 1 and 15. Value used: 0")



if __name__ == "__main__":
    pyunit_utils.standalone_test(unittest.main)
else:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestXGBoostMaxDepth)
    unittest.TextTestRunner().run(suite)
