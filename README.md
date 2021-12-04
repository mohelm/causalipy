# py-metrics

Econometric tools for Python


## TODO

## Large

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

## Small

- Ask Simon about where to put the app
- What plotting libary to use -> probably Seaborn (in the app it would of course
  be better to use Altair or Plotly). Probably you could switch between the
  graphing libraries relatively easily and start with Seaborn.
- Add aggregate effects
- Add "zero effects" - why does this not work with your library
- Do not hardcode column names
- Clean up OLS and LogisticRegression files 
- Add mypy, flake8 and black plugins to pytest - do you even want to do this?
- Github actions pipeline (especially relevant for the app) - but maybe do one
  for tests here as well.


