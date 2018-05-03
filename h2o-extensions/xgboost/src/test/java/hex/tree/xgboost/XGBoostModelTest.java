package hex.tree.xgboost;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import water.H2O;

import static org.junit.Assert.*;

public class XGBoostModelTest {

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void testCreateParamsNThreads() throws Exception {
    // default
    XGBoostModel.XGBoostParameters pDefault = new XGBoostModel.XGBoostParameters();
    pDefault._backend = XGBoostModel.XGBoostParameters.Backend.cpu; // to disable the GPU check
    BoosterParms bpDefault = XGBoostModel.createParams(pDefault, 2);
    assertEquals((int) H2O.ARGS.nthreads, bpDefault.get().get("nthread"));
    // user specified
    XGBoostModel.XGBoostParameters pUser = new XGBoostModel.XGBoostParameters();
    pUser._backend = XGBoostModel.XGBoostParameters.Backend.cpu; // to disable the GPU check
    pUser._nthread = H2O.ARGS.nthreads - 1;
    BoosterParms bpUser = XGBoostModel.createParams(pUser, 2);
    assertEquals((int) H2O.ARGS.nthreads - 1, bpUser.get().get("nthread"));
    // user specified (over the limit)
    XGBoostModel.XGBoostParameters pOver = new XGBoostModel.XGBoostParameters();
    pOver._backend = XGBoostModel.XGBoostParameters.Backend.cpu; // to disable the GPU check
    pOver._nthread = H2O.ARGS.nthreads + 1;
    BoosterParms bpOver = XGBoostModel.createParams(pOver, 2);
    assertEquals((int) H2O.ARGS.nthreads, bpOver.get().get("nthread"));
  }

  @Test
  public void testMaxDepthLowerBound() {
    // Max depth must be in <1,15> (inclusive interval)
    // Testing the lower bound
    XGBoostModel.XGBoostParameters parameters = new XGBoostModel.XGBoostParameters();
    parameters._max_depth = 0;

    expectedException.expect(UnsupportedOperationException.class);
    expectedException.expectMessage("MAX_DEPTH limit for XGBoost must be between 1 and 15. Value used: 0");
    XGBoostModel.createParams(parameters, 2);
  }

  @Test
  public void testMaxDepthUpperBound() {
    // Max depth must be in <1,15> (inclusive interval)
    // Testing the upper bound
    XGBoostModel.XGBoostParameters parameters = new XGBoostModel.XGBoostParameters();
    parameters._max_depth = 16;

    expectedException.expect(UnsupportedOperationException.class);
    expectedException.expectMessage("MAX_DEPTH limit for XGBoost must be between 1 and 15. Value used: 16");
    XGBoostModel.createParams(parameters, 2);
  }

}