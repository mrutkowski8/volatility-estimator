# **PURPOSE**

#### i made this project to see how S&P 500 volatility looks when you measure it in different ways

* calculate annualized volatility over the full sample and over rolling windows of 20, 60, and 120 trading sessions
* compare the rolling estimates with EWMA, which gives more weight to recent returns
* compare volatility over the five trading sessions before and after each CPI or FOMC release

the program uses the first NYSE close after a release to identify the event session. it reports that session's return separately.

# **REQUIREMENTS**

#### install the libraries from the project folder

```bash
python -m pip install -r requirements.txt
```

# **DATA**

#### to try it out, generate a set of synthetic closing prices

```bash
python make_demo.py
```

this creates `data/demo.csv`. the included `data/events.csv` has four example CPI and FOMC releases.

to use your own data, supply a CSV with `observation_date` and `SP500` columns and a closing price for every NYSE session in your date range. use `--start`, `--end`, and `--events` to select the dates and release calendar you want to analyze.

# **RUN**

#### run the program with the demo data

```bash
python volatility.py data/demo.csv --source-label synthetic
```

the results go into `output/demo/`

* `volatility.png` shows the rolling estimates and EWMA on the same chart
* `volatility.csv` contains the daily returns and volatility estimates
* `event_study.csv` shows the before-and-after comparison for each release
* `run_metadata.json` records the inputs and settings used for the run

*if you're trying the demo, look at how quickly EWMA and the 120-session estimate react to the same simulated drop.*
