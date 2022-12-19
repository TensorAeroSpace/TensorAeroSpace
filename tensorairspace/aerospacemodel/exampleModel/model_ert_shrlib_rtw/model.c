/*
 * File: model.c
 *
 * Code generated for Simulink model 'model'.
 *
 * Model version                  : 1.41
 * Simulink Coder version         : 8.9 (R2015b) 13-Aug-2015
 * C/C++ source code generated on : Sun Dec 18 21:44:12 2022
 *
 * Target selection: ert_shrlib.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "model.h"
#include "model_private.h"

/* Block signals (auto storage) */
B_model_T model_B;

/* Continuous states */
X_model_T model_X;

/* External inputs (root inport signals with auto storage) */
ExtU_model_T model_U;

/* External outputs (root outports fed by signals with auto storage) */
ExtY_model_T model_Y;

/* Real-time model */
RT_MODEL_model_T model_M_;
RT_MODEL_model_T *const model_M = &model_M_;

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  /* Solver Matrices */
  static const real_T rt_ODE3_A[3] = {
    1.0/2.0, 3.0/4.0, 1.0
  };

  static const real_T rt_ODE3_B[3][3] = {
    { 1.0/2.0, 0.0, 0.0 },

    { 0.0, 3.0/4.0, 0.0 },

    { 2.0/9.0, 1.0/3.0, 4.0/9.0 }
  };

  time_T t = rtsiGetT(si);
  time_T tnew = rtsiGetSolverStopTime(si);
  time_T h = rtsiGetStepSize(si);
  real_T *x = rtsiGetContStates(si);
  ODE3_IntgData *id = (ODE3_IntgData *)rtsiGetSolverData(si);
  real_T *y = id->y;
  real_T *f0 = id->f[0];
  real_T *f1 = id->f[1];
  real_T *f2 = id->f[2];
  real_T hB[3];
  int_T i;
  int_T nXc = 8;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) memcpy(y, x,
                (uint_T)nXc*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  model_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  model_step();
  model_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  model_step();
  model_derivatives();

  /* tnew = t + hA(3);
     ynew = y + f*hB(:,3); */
  for (i = 0; i <= 2; i++) {
    hB[i] = h * rt_ODE3_B[2][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
  }

  rtsiSetT(si, tnew);
  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

/* Model step function */
void model_step(void)
{
  real_T u0;
  if (rtmIsMajorTimeStep(model_M)) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&model_M->solverInfo,((model_M->Timing.clockTick0+1)*
      model_M->Timing.stepSize0));
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(model_M)) {
    model_M->Timing.t[0] = rtsiGetT(&model_M->solverInfo);
  }

  /* Integrator: '<Root>/Integrator1' */
  model_B.Integrator1 = model_X.Integrator1_CSTATE;

  /* Outport: '<Root>/Wz' */
  model_Y.Wz = model_B.Integrator1;

  /* Outport: '<Root>/theta_big' incorporates:
   *  Integrator: '<Root>/Integrator4'
   */
  model_Y.theta_big = model_X.Integrator4_CSTATE;

  /* Outport: '<Root>/H' incorporates:
   *  Integrator: '<Root>/Integrator3'
   */
  model_Y.H = model_X.Integrator3_CSTATE;

  /* Sum: '<Root>/Sum4' incorporates:
   *  Inport: '<Root>/refSignal'
   *  Integrator: '<Root>/Integrator'
   */
  model_Y.alpha = model_U.refSignal + model_X.Integrator_CSTATE;

  /* Outport: '<Root>/theta_small ' incorporates:
   *  Integrator: '<Root>/Integrator2'
   */
  model_Y.theta_small = model_X.Integrator2_CSTATE;

  /* Integrator: '<S1>/Integrator2' */
  /* Limited  Integrator  */
  if (model_X.Integrator2_CSTATE_j >= 20.0) {
    model_X.Integrator2_CSTATE_j = 20.0;
  } else {
    if (model_X.Integrator2_CSTATE_j <= -25.0) {
      model_X.Integrator2_CSTATE_j = -25.0;
    }
  }

  /* Sum: '<Root>/Sum3' incorporates:
   *  Gain: '<Root>/Gain2'
   *  Gain: '<Root>/Gain9'
   *  Integrator: '<S1>/Integrator2'
   */
  model_B.Sum3 = 0.0993 * model_X.Integrator2_CSTATE_j + 0.689 * model_Y.alpha;

  /* Sum: '<Root>/Sum5' incorporates:
   *  Inport: '<Root>/refSignal'
   *  Integrator: '<Root>/Integrator2'
   */
  model_B.Sum5 = model_U.refSignal + model_X.Integrator2_CSTATE;

  /* Gain: '<Root>/Gain6' incorporates:
   *  Integrator: '<Root>/Integrator4'
   */
  model_B.Gain6 = 3.86 * model_X.Integrator4_CSTATE;

  /* Sum: '<Root>/Sum2' */
  model_B.Sum2 = model_B.Integrator1 - model_B.Sum3;

  /* TransferFcn: '<S1>/Transfer Fcn' */
  model_B.Saturation = 833.33333333333337 * model_X.TransferFcn_CSTATE;

  /* Saturate: '<S1>/Saturation' */
  if (model_B.Saturation > 25.0) {
    model_B.Saturation = 25.0;
  } else {
    if (model_B.Saturation < -25.0) {
      model_B.Saturation = -25.0;
    }
  }

  /* End of Saturate: '<S1>/Saturation' */

  /* Sum: '<Root>/Sum' incorporates:
   *  Gain: '<Root>/Gain10'
   *  Gain: '<Root>/Gain11'
   *  Gain: '<Root>/Gain12'
   *  Gain: '<Root>/Gain3'
   *  Gain: '<Root>/Gain4'
   *  Sum: '<Root>/Sum6'
   *  TransferFcn: '<Root>/Transfer Fcn1'
   */
  u0 = ((0.1 * model_X.TransferFcn1_CSTATE + 0.024 * model_B.Sum5) * 1.3 + 0.36 *
        model_B.Sum3 * 2.2) + 0.4 * model_B.Integrator1;

  /* Saturate: '<S1>/Saturation1' */
  if (u0 > 20.0) {
    u0 = 20.0;
  } else {
    if (u0 < -25.0) {
      u0 = -25.0;
    }
  }

  /* Sum: '<S1>/Sum' incorporates:
   *  Integrator: '<S1>/Integrator2'
   *  Saturate: '<S1>/Saturation1'
   */
  model_B.Sum = u0 - model_X.Integrator2_CSTATE_j;

  /* Sum: '<Root>/Sum1' incorporates:
   *  Gain: '<Root>/Gain1'
   *  Gain: '<Root>/Gain5'
   *  Gain: '<Root>/Gain7'
   *  Gain: '<Root>/Gain8'
   *  Inport: '<Root>/refSignal'
   *  Integrator: '<S1>/Integrator2'
   */
  model_B.Sum1 = (((-16.45 * model_X.Integrator2_CSTATE_j + model_U.refSignal) +
                   -8.98 * model_Y.alpha) + -0.196 * model_B.Sum2) + -1.028 *
    model_B.Integrator1;
  if (rtmIsMajorTimeStep(model_M)) {
    rt_ertODEUpdateContinuousStates(&model_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++model_M->Timing.clockTick0;
    model_M->Timing.t[0] = rtsiGetSolverStopTime(&model_M->solverInfo);

    {
      /* Update absolute timer for sample time: [0.1s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.1, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       */
      model_M->Timing.clockTick1++;
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void model_derivatives(void)
{
  boolean_T lsat;
  boolean_T usat;
  XDot_model_T *_rtXdot;
  _rtXdot = ((XDot_model_T *) model_M->ModelData.derivs);

  /* Derivatives for Integrator: '<Root>/Integrator1' */
  _rtXdot->Integrator1_CSTATE = model_B.Sum1;

  /* Derivatives for Integrator: '<Root>/Integrator4' */
  _rtXdot->Integrator4_CSTATE = model_B.Sum3;

  /* Derivatives for Integrator: '<Root>/Integrator3' */
  _rtXdot->Integrator3_CSTATE = model_B.Gain6;

  /* Derivatives for Integrator: '<Root>/Integrator' */
  _rtXdot->Integrator_CSTATE = model_B.Sum2;

  /* Derivatives for Integrator: '<Root>/Integrator2' */
  _rtXdot->Integrator2_CSTATE = model_B.Integrator1;

  /* Derivatives for Integrator: '<S1>/Integrator2' */
  lsat = (model_X.Integrator2_CSTATE_j <= -25.0);
  usat = (model_X.Integrator2_CSTATE_j >= 20.0);
  if (((!lsat) && (!usat)) || (lsat && (model_B.Saturation > 0.0)) || (usat &&
       (model_B.Saturation < 0.0))) {
    _rtXdot->Integrator2_CSTATE_j = model_B.Saturation;
  } else {
    /* in saturation */
    _rtXdot->Integrator2_CSTATE_j = 0.0;
  }

  /* End of Derivatives for Integrator: '<S1>/Integrator2' */

  /* Derivatives for TransferFcn: '<Root>/Transfer Fcn1' */
  _rtXdot->TransferFcn1_CSTATE = 0.0;
  _rtXdot->TransferFcn1_CSTATE += -0.0 * model_X.TransferFcn1_CSTATE;
  _rtXdot->TransferFcn1_CSTATE += model_B.Sum5;

  /* Derivatives for TransferFcn: '<S1>/Transfer Fcn' */
  _rtXdot->TransferFcn_CSTATE = 0.0;
  _rtXdot->TransferFcn_CSTATE += -33.333333333333336 *
    model_X.TransferFcn_CSTATE;
  _rtXdot->TransferFcn_CSTATE += model_B.Sum;
}

/* Model initialize function */
void model_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)model_M, 0,
                sizeof(RT_MODEL_model_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&model_M->solverInfo, &model_M->Timing.simTimeStep);
    rtsiSetTPtr(&model_M->solverInfo, &rtmGetTPtr(model_M));
    rtsiSetStepSizePtr(&model_M->solverInfo, &model_M->Timing.stepSize0);
    rtsiSetdXPtr(&model_M->solverInfo, &model_M->ModelData.derivs);
    rtsiSetContStatesPtr(&model_M->solverInfo, (real_T **)
                         &model_M->ModelData.contStates);
    rtsiSetNumContStatesPtr(&model_M->solverInfo, &model_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&model_M->solverInfo,
      &model_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&model_M->solverInfo,
      &model_M->ModelData.periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&model_M->solverInfo,
      &model_M->ModelData.periodicContStateRanges);
    rtsiSetErrorStatusPtr(&model_M->solverInfo, (&rtmGetErrorStatus(model_M)));
    rtsiSetRTModelPtr(&model_M->solverInfo, model_M);
  }

  rtsiSetSimTimeStep(&model_M->solverInfo, MAJOR_TIME_STEP);
  model_M->ModelData.intgData.y = model_M->ModelData.odeY;
  model_M->ModelData.intgData.f[0] = model_M->ModelData.odeF[0];
  model_M->ModelData.intgData.f[1] = model_M->ModelData.odeF[1];
  model_M->ModelData.intgData.f[2] = model_M->ModelData.odeF[2];
  model_M->ModelData.contStates = ((X_model_T *) &model_X);
  rtsiSetSolverData(&model_M->solverInfo, (void *)&model_M->ModelData.intgData);
  rtsiSetSolverName(&model_M->solverInfo,"ode3");
  rtmSetTPtr(model_M, &model_M->Timing.tArray[0]);
  model_M->Timing.stepSize0 = 0.1;

  /* block I/O */
  (void) memset(((void *) &model_B), 0,
                sizeof(B_model_T));

  /* states (continuous) */
  {
    (void) memset((void *)&model_X, 0,
                  sizeof(X_model_T));
  }

  /* external inputs */
  model_U.refSignal = 0.0;

  /* external outputs */
  (void) memset((void *)&model_Y, 0,
                sizeof(ExtY_model_T));

  /* InitializeConditions for Integrator: '<Root>/Integrator1' */
  model_X.Integrator1_CSTATE = 0.0;

  /* InitializeConditions for Integrator: '<Root>/Integrator4' */
  model_X.Integrator4_CSTATE = 0.0;

  /* InitializeConditions for Integrator: '<Root>/Integrator3' */
  model_X.Integrator3_CSTATE = 0.0;

  /* InitializeConditions for Integrator: '<Root>/Integrator' */
  model_X.Integrator_CSTATE = 0.0;

  /* InitializeConditions for Integrator: '<Root>/Integrator2' */
  model_X.Integrator2_CSTATE = 0.0;

  /* InitializeConditions for Integrator: '<S1>/Integrator2' */
  model_X.Integrator2_CSTATE_j = 0.0;

  /* InitializeConditions for TransferFcn: '<Root>/Transfer Fcn1' */
  model_X.TransferFcn1_CSTATE = 0.0;

  /* InitializeConditions for TransferFcn: '<S1>/Transfer Fcn' */
  model_X.TransferFcn_CSTATE = 0.0;
}

/* Model terminate function */
void model_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
