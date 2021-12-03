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

Substantial extensions

- Add aggregate effects

- Add the data (and update the README) 5

  - Can you do it like Seaborn

- Simulation function finish it and test it 5

Refactoring:

- Finish refactoring of classes 5
  - Test the data handler.
- Clean up OLS and LogisticRegression files 5

README

- Add installation instructions (and try them out) 5
- Update the graph 5

- Add mypy, flake8 and black plugins to pytest - do you even want to do this?
- Github actions pipeline (especially relevant for the app) - but maybe do one
  for tests here as well.

- Think of better names for the DR estimator

### Large

Demo app:

- Think about where to put the app
