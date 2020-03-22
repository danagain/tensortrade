import ssl
import pandas as pd
from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.data import Stream, DataFeed, Module
from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.environments import TradingEnvironment
from tensortrade.agents import DQNAgent
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives a SSLError

def fetch(exchange_name, symbol, timeframe):
    url = "https://www.cryptodatadownload.com/cdd/"
    filename = "{}_{}USD_{}.csv".format(exchange_name, symbol, timeframe)
    volume_column = "Volume {}".format(symbol)
    new_volume_column = "Volume_{}".format(symbol)
    
    df = pd.read_csv(url + filename, skiprows=1)
    df = df[::-1]
    df = df.drop(["Symbol"], axis=1)
    df = df.rename({"Volume USD": "volume", volume_column: new_volume_column}, axis=1)
    df = df.set_index("Date")
    df.columns = [symbol + ":" + name.lower() for name in df.columns]
                     
    return df


coinbase_data = pd.concat([
    fetch("Coinbase", "BTC", "1h"),
    # fetch("Coinbase", "ETH", "1h")
], axis=1)

coinbase = Exchange("coinbase", service=execute_order)(
    Stream("USD-BTC", list(coinbase_data['BTC:close'])),
    # Stream("USD-ETH", list(coinbase_data['ETH:close']))
)

coinbase_btc = coinbase_data.loc[:, [name.startswith("BTC") for name in coinbase_data.columns]]

with Module("coinbase") as coinbase_ns:
    nodes = [Stream(name, list(coinbase_data[name])) for name in coinbase_data.columns]

feed = DataFeed([coinbase_ns])

portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 1 * BTC),
    # Wallet(coinbase, 5 * ETH),
])


env = TradingEnvironment(
    feed=feed,
    portfolio=portfolio,
    action_scheme='managed-risk',
    reward_scheme='risk-adjusted',
    window_size=20
)


agent = DQNAgent(env)

agent.train(n_steps=24000, save_path="examples/agents/", n_episodes=1)

portfolio.performance.plot()
plt.show()
portfolio.performance.net_worth.plot()
plt.show()