# CausaliPy

Causal Methods implemented in Python.

## Installation

## Example

To run a version of the multi-period difference-in-difference estimator as
proposed by Callaway and Sant’Anna (2020)  (this requires additionally pyarrow  - e.g. via
`pip install pyarrow` - to be installed currently):

```python
url = "https://github.com/mohelm/causalipy-datasets/raw/main/mpdta-sample.feather"
data = pd.read_feather(url)

mpd_minimum_wage = MultiPeriodDid(
    data,
    outcome="lemp",
    treatment_indicator="treat",
    time_period_indicator="year",
    group_indiciator="first.treat",
    formula="~ 1",
)
mpd_minimum_wage.plot_treatment_effects()
```

This will give:

![alt text](./readme_fig.png)

## License

This project is licensed under the terms of the MIT license.

