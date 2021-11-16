Calibration and Simulation Engine for Local Volatility Models
-------------------------------------------------------------

This repository contains calibration and simulation routines for deterministic/stochastic local volatility models with deterministic/stochastic interest rates, implemented in Python.

The manuscript that gives the details of the calibration algorithms can be accessed at https://arxiv.org/abs/2009.14764. This repository provides a sample implementation of these algorithms.

The repository contains implementation of calibration of local volaility surface for EUR/USD currency pair though the example is easily adapted to other currency pairs with appropriately calibrated G1pp parameters and discount curves.

The repository contains notebooks detailing:
- The calibration procedure 
- Visualization of local volatility surface
- Repricing with the calibrated local volality surface
- Comparing repriced vs market implied volatility

For three different cases:
- Local volatility surface with 2 (Domestic and Foreign) stochastic interest rates (LV2SR)
- Stochastic Local volatility with 2 (Domestic and Foreign) deterministic interest rates (SLV2DR)
- Stochastic Local volatility with 2 (Domestic and Foreign) stochastic interest rates (SLV2SR)

# Files:
- **marketdata_JSON_asof_04_30_2020/**: Market data in JSON format containing EUR and USD discount curves. Risk-Reversal and Butterfly calibrated market prices and implied volatility for EUR/USD.
  - **EURUSD_Heston.json**: Heston parameterization of Stochastic Local Volatility for EUR/USD.

- **lib/**: library and utilities for reading and interpolating market data as well as calibration routines.
  - **bsanalytic.py**: Black-Scholes analytic formulas for relevant market instruments
  - **calibrator.py**: Main utilities for calibrating local volatility surface. Classes include local volatility calibration under Call surface or Total implied variance (TIV) formulation
  - **fxivolinterpolator.py**: Utilities for interpolating time-discretized and strike-discretized market data for implied volatility.
  - **surfaces.py**: Classes and utilities for constructing Call surface and TIV surface needed for calibration.
  - **interpolator.py**: Classes and utilities for interpolating constructed TIV and Call surface needed for calibration.

- **sim_lib/StochasticSim_Multiprocessing.py**: Numpy based local volality and stochastic local volatility simulator. This simulates assets, interest rates, and volatility under the given local or stochastic local volatility and interest rates.
  - **Assets'** time-evolution described as GBM.
  - **Stochastic Interest rates** parametrized and described as Hull-White/G1pp processes
  - **Stochastic Local Volatility** described by Heston Model.

- **LV_2SIR/FX_LocalVol_Calibration.ipynb**: Notebook demonstrating calibration of LV_2SIR and Repricing under Local Volatility.

- **/SLV_2DIR/**:
  - **Calibrate_SLV_2DIR.ipynb**: Notebook demonstrating calibration of Leverage surface of the Stochastic Local volatility under deterministic rates.
  - **Reprice_SLV_2DIR.ipynb**: Reprice under the calibrated stochastic local volatility model under determistic rates.

- **/SLV_2SIR/**:
  - **Calibrate_SLV_2DIR.ipynb**: Notebook demonstrating calibration of Leverage surface of the Stochastic Local volatility under stochastic rates.
  - **Reprice_SLV_2DIR.ipynb**: Reprice under the calibrated stochastic local volatility model under stochastic rates.

## Usage:
Clone the repository. With an installation of Jupyter with Python kernel >=3.6, run notebooks in folders **/LV_2SIR, /SLV_2DIR and /SLV_2SIR** for the Local Volatility with stochastic rates, Stochastic Local volatlity with deterministic rates and Stochastic Local volatility with Stochastic rates respectively.

# Overview of the model and Summary of Results

## The model (LV2SR)
This corresponding model for the underlying FX process with 2 (domestic and foreign) rate is assumed to be: 

<img src="https://render.githubusercontent.com/render/math?math=\Large{dS_t = \left[r^d_t - r^f_t \right] S_t dt %2B \sigma(S_t, t) S_t dW^{S^\text{DRN}}_t}">

where the domestic and foreign rates are parametrized using the G1pp model. The domestic rate evolves in the domestic risk neutral measure as 

<img src="https://render.githubusercontent.com/render/math?math=\Large{r^d_t = x^d_t %2B \phi^d_t}"><br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dx^d_t = -a^d_t x^d_t dt %2B \sigma^d_t dW^{d\text{(DRN)}}_t}"><br>

whereas the foreign short rate evolves in foreign risk neutral measure,

<img src="https://render.githubusercontent.com/render/math?math=\Large{r^f_t = x^f_t %2B \phi^f_t}"><br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dx^f_t = -a^f_t x^f_t dt %2B \sigma^f_t dW^{f\text{(FRN)}}_t}"><br>

## Calibration of local volatility surface (LV2SR)
The local volality surface or state dependent diffusion coefficient <img src="https://render.githubusercontent.com/render/math?math=\sigma(S_t, t)"> is calibrated in the domestic **T-Forward measure**. The procedure is as follows:

- The calibration is performed in a time slice-by-slice basis.
- The underling FX model and the domestic and foreign rates are first simulated in the T-Fwd measure with the local volatility until current time slice.
- The expectation <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}^{\mathbb{Q}^{\text{T}}}\left[(K r_T^d - S_T r_T^f) \mathbb{1}_{S_T > K}\right]"> is gathered from the T-Fwd simulation of the interest rates and underliers.
- The following extension to Dupire's formula for stochastic rates is used to compute the local volatility at the current time slice.

<img src="https://render.githubusercontent.com/render/math?math=\huge{\sigma_{\text{LV (stochastic rates)}}^2 = \frac{\frac{\partial C_{\text{BS}}}{\partial T}- P^d(0, T) \mathbb{E}^{Q^{\text{T}}}\left[(K r_T^d - S_T r_T^f) \mathbb{1}_{S_T > K}\right]}{\frac{\partial C_{\text{BS}}}{\partial w} \left[1 - \frac{y}{w} \frac{\partial w}{\partial y} %2B \frac{1}{2} \frac{\partial^2 w}{\partial y^2} %2B \frac{1}{4} \left(\frac{\partial w}{\partial y}\right)^2\left(-\frac{1}{4}- \frac{1}{w} %2B \frac{y^2}{w^2}\right)\right]}}">

- The procedure is repeated for all time slices starting from 0 to maturity.

where: 
<img src="https://render.githubusercontent.com/render/math?math=T">: Time to maturity 

<img src="https://render.githubusercontent.com/render/math?math=C_{\text{BS}}"> : The Black-Scholes call price

<img src="https://render.githubusercontent.com/render/math?math=w=\sigma_{\text{BS}}T^2"> : Total implied variance

<img src="https://render.githubusercontent.com/render/math?math=S_T">: Value of Underlier at time.

<img src="https://render.githubusercontent.com/render/math?math=K">: Strike

<img src="https://render.githubusercontent.com/render/math?math=y=\log\left(\frac{S_T}{K}\right)"> : Log-moneyness

Local volatility surface obtained via different number of monte-carlo paths.
![LV_2SR_Convergence](https://user-images.githubusercontent.com/12563351/141022625-c281469d-dd3e-4bf9-94cb-c31cd48ddac7.png)

Call Option price and implied volatility Recovery:
![LV_2SR_maturity_diff_call_and_ivol](https://user-images.githubusercontent.com/12563351/141033827-4e3c9b81-1911-4a50-9c1e-77d73a61bc62.png)


## Stochastic Local Volatility with 2 Deterministic Rates (SLV2DR)
## The model.

The model of stochastic local volatility is modeled as a CIR(Cox-Ingersoll-Ross) process.

<img src="https://render.githubusercontent.com/render/math?math=\Large{dS_t=\left[r^d_t - r^f_t\right] S_t dt %2B L(S_t, t) \sqrt{U_t} S_t dW_t^{S\text{(DRN)}}}">

<img src="https://render.githubusercontent.com/render/math?math=\Large{dU_t=\kappa_t (\theta_t - U_t) dt %2B \xi_t \sqrt{U_t} dW_t^{U\text{(DRN)}}}">

where: 
<img src="https://render.githubusercontent.com/render/math?math=\Large{S_t}">: Underlier's value at time

<img src="https://render.githubusercontent.com/render/math?math=\Large{r^d_t, r^f_t}"> : The corresponding domestic and foreign rates (deterministic)

<img src="https://render.githubusercontent.com/render/math?math=\Large{L(S_t, t)}"> : State dependent leverage function

<img src="https://render.githubusercontent.com/render/math?math=\Large{U_t}">: The variance process

<img src="https://render.githubusercontent.com/render/math?math=\Large{\kappa_t, \theta_t, \xi_t}">: The corresponding mean reversion, long-term variance and vol-of-vol parameters of the Heston process.

<img src="https://render.githubusercontent.com/render/math?math=\Large{dW_t^{U\text{(DRN)}}, dW_t^{S\text{(DRN)}}}"> : Browninan drivers of the underlier and variance processes in Domestic Risk neutral measure


## Calibration of the Leverage Surface

The leverage function is related to the local volatility calibrated with deterministic rates via the Dupire's formula and the expectation of variance process as:

<img src="https://render.githubusercontent.com/render/math?math=\Large{\sigma_{\text{LV}}(x, t)^2 = L(x, t)^2 \mathbf{E}^{\mathbb{Q}^{\text{(DRN)}}}\left[U_t \mid S_t=x \right]}">

- The conditional expecation <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}^{\mathbb{Q}^{\text{(DRN)}}}\left[U_t \mid S_t=x \right]"> can be computed by binning the underlier values at time from the simulation sample paths as a function of <img src="https://render.githubusercontent.com/render/math?math=S_t, t"> in T-Fwd measure.
- Alternatively, the expectation can also obtained by regressing on the risk-factor, (here <img src="https://render.githubusercontent.com/render/math?math=S_t">) as described in the paper.
- The value of <img src="https://render.githubusercontent.com/render/math?math=\sigma_{\text{LV}}(x, t)^2"> is computed from deterministic Dupire's formula.
- Finally the value of leverage function <img src="https://render.githubusercontent.com/render/math?math=L(x, t)^2"> on the grid is obtained by dividing <img src="https://render.githubusercontent.com/render/math?math=\sigma_{\text{LV}}(x, t)^2"> by <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}^{\mathbb{Q}^{\text{(DRN)}}}\left[U_t \mid S_t=x \right]">


The leverage function calibrated with deterministic rates with different number of monte-carlo calibration paths.
![SLV_2DR_Convergence](https://user-images.githubusercontent.com/12563351/141051960-bcf55031-9cd1-419f-87b6-87a48d1588e2.png)

The repriced call function at maturity and the corresponding implied vol recovered within +-2 Monte Carlo errors.
![SLV_2DR_maturity_diff_call_and_ivol](https://user-images.githubusercontent.com/12563351/141052061-d024c83f-deb1-4b1e-bc81-331684bdaa20.png)

## Stochastic Local Volatility with 2 Stochastic Rates (SLV2SR)
## The model.
The model which is a mixture of stochastic local volatility with stochastic rates can be written as:

The underlier modeled as CIR dynamics:<br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dS_t = \left[r^d_t - r^f_t\right] S_t dt %2B L(S_t, t)\sqrt{U_t} S_t dW_t^{S\text{(DRN)}}}"><br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dU_t = \kappa_t (\theta_t - U_t) dt %2B \xi_t \sqrt{U_t} dW_t^{U\text{(DRN)}}}"><br>

Domestic rates modeled as a G1pp process:<br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{r^d_t = x^d_t %2B \phi^d_t}"><br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dx^d_t = -a^d_t x^d_t dt %2B \sigma^d_t dW_t^{d\text{(DRN)}}}"><br>

Foreign rates modeled as a second G1pp process:<br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{r^f_t = x^f_t %2B \phi^f_t}"><br>
<img src="https://render.githubusercontent.com/render/math?math=\Large{dx^f_t = \left[-a^f_t x^f_t - \rho_{Sf} \sigma^f_t L(S_t, t) \sqrt{U_t}\right] dt %2B \sigma^f_t dW_t^{f\text{(DRN)}}}"><br>

## Calibration of the leverage surface

The calibrated leverage function is related to the local volatility (LV2SR) and the variance process by:
<img src="https://render.githubusercontent.com/render/math?math=\Large{\sigma_{\text{LV}}(x, t)^2 = L(x, t)^2 \mathbf{E}^{\mathbb{Q}^{\text{T}}} \left[U_t \mid S_t=x \right]}">

- The conditional expecation <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}^{\mathbb{Q}^{\text{(DRN)}}}\left[U_t \mid S_t=x \right]"> can be computed by binning the underlier values at time from the simulation sample paths as a function of <img src="https://render.githubusercontent.com/render/math?math=S_t, t"> in T-Fwd measure.
- Alternatively, the expectation can also obtained by regressing on the risk-factor, (here <img src="https://render.githubusercontent.com/render/math?math=S_t">) as described in the paper.
- The value of <img src="https://render.githubusercontent.com/render/math?math=\sigma_{\text{LV}}(x, t)^2"> is computed as in the LV2SIR method.
- Finally the value of leverage function <img src="https://render.githubusercontent.com/render/math?math=L(x, t)^2"> on the grid is obtained by dividing <img src="https://render.githubusercontent.com/render/math?math=\sigma_{\text{LV}}(x, t)^2"> by <img src="https://render.githubusercontent.com/render/math?math=\mathbf{E}^{\mathbb{Q}^{\text{(DRN)}}}\left[U_t \mid S_t=x \right]">


The repriced Call option prices and the implied volatility recovered with the repriced options fall within +- 2 MC errors.
![SLV_2SR_maturity_diff_call_and_ivol](https://user-images.githubusercontent.com/12563351/141054532-cb876819-1bf7-4729-b88e-6bdfc7113c50.png)
