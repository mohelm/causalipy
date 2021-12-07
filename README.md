# CausaliPy

Causal Methods implemented in Python.

## Installation

## Example

To run a version of the multi-period difference-in-difference estimator as
proposed by Callaway and Sant’Anna (2020):

```python
data = pd.read_feather(data)
mpd_minimum_wage = MultiPeriodDid(
    data,
    outcome="outcome",
    treatment_indicator="treatment_status",
    formula_or="outcome ~ 1",
    formula_ipw="treatment_status ~ 1",
)
mpd_minimum_wage.plot_treatment_effects()
```

This will give:

![alt text](./readme_fig.png)

## License

This project is licensed under the terms of the MIT license.

## TODO

### Small

Demo app:

- Think about where to put the app

Substantial extensions

- Add aggregate effects

Clean-up

- Clean up OLS and LogisticRegression files
- Add mypy, flake8 and black plugins to pytest - do you even want to do this?
- Github actions pipeline (especially relevant for the app) - but maybe do one
  for tests here as well.

- Update this README:
  - Installation

### Large

- The main disadvantages is currently that the or estimator and the ipw
  estimator are deeply hidden inside the DR estimator. It would be better to
  have the DR estimator be somehow composed out of the DR estimator and the IPW
  estimator.

  - Maybe think about it like that - you have the data and then you have the two
    models (or and ipw), and they somehow combine the data with these estimates.
    Maybe you need three entities - one which handles the data, one which
    handles ipw and one which handles or. The data class mainly has to return
    the treatment status and the outcome.
  - You could try that first with the ATT which is much simpler.
  - The DR class is then mainly composed of the two estimatators and maybe also
    the data class.

- In the shorter term, however, it would be good to do some code clean-ups and
  move the interface towards useability.

- Interesting inbetween move could be to move the estimator implementation
  closer to the paper...
