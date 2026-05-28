import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import io

st.set_page_config(page_title="Crypto Spot Backtest", layout="wide")

# =========================
# FILE UPLOADER
# =========================
uploaded_files = st.file_uploader(
    "Upload Crypto CSV Files",
    accept_multiple_files=True,
    type=['csv']
)

# =========================
# SIDEBAR
# =========================
with st.sidebar.form("config_form"):

    st.header("⚙️ Crypto Spot Backtest")

    INITIAL_CAPITAL = st.number_input(
        "Initial Capital",
        min_value=100,
        value=10000,
        step=1000
    )

    MAX_POSITION_SIZE = st.number_input(
        "Max Position Size",
        min_value=10,
        value=1000,
        step=100
    )

    fee_rate = st.slider(
        "Trading Fee (%)",
        0.0,
        1.0,
        0.1,
        step=0.01
    ) / 100

    start_date = st.date_input(
        "Start Date",
        value=date(2020, 1, 1)
    )

    end_date = st.date_input(
        "End Date",
        value=date.today()
    )

    nen_tich_luy = st.slider(
        "Base Max Range %",
        0.01,
        0.20,
        0.05,
        step=0.01
    )

    min_days = st.number_input(
        "Min Base Days",
        value=4
    )

    max_days = st.number_input(
        "Max Base Days",
        value=10
    )

    breakout_days_check = st.number_input(
        "Breakout Days Check",
        value=3
    )

    max_chase = st.slider(
        "Max Chase",
        1.0,
        1.2,
        1.05,
        step=0.01
    )

    target = st.slider(
        "Take Profit",
        1.0,
        3.0,
        1.4,
        step=0.05
    )

    stoploss = st.slider(
        "Stoploss",
        0.5,
        1.0,
        0.93,
        step=0.01
    )

    min_hold_days = st.number_input(
        "Min Hold Candles",
        value=0
    )

    max_hold_days = st.number_input(
        "Max Hold Candles",
        value=200
    )

    avoid_duplicate = st.number_input(
        "Avoid Duplicate Candles",
        value=5
    )

    min_volume_ratio = st.slider(
        "Min Volume Ratio",
        0.5,
        10.0,
        1.5,
        step=0.1
    )

    max_volume_ratio = st.slider(
        "Max Volume Ratio",
        1.0,
        20.0,
        8.0,
        step=0.5
    )

    use_atr = st.checkbox("Use ATR Stoploss", value=False)

    atr_multiplier = st.slider(
        "ATR Multiplier",
        0.5,
        5.0,
        2.0,
        step=0.1
    )

    run_backtest_button = st.form_submit_button("🚀 RUN BACKTEST")

# =========================
# VALIDATE DATE
# =========================
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date >= end_date:
    st.error("Start date must be smaller than end date")
    st.stop()

# =========================
# CONFIG
# =========================
so_ngay_tich_luy = range(int(min_days), int(max_days) + 1)

BACKTEST_CONFIG = {
    "nen_tich_luy": nen_tich_luy,
    "so_ngay_tich_luy": so_ngay_tich_luy,
    "breakout_days_check": int(breakout_days_check),
    "max_chase": max_chase,
    "target": target,
    "stoploss": stoploss,
    "min_hold_days": int(min_hold_days),
    "max_hold_days": int(max_hold_days),
    "avoid_duplicate": int(avoid_duplicate),
    "min_volume_ratio": min_volume_ratio,
    "max_volume_ratio": max_volume_ratio,
    "use_atr": use_atr,
    "atr_multiplier": atr_multiplier
}

# =========================
# READ FILE
# =========================
def read_crypto_file(file_wrapper):

    df = pd.read_csv(file_wrapper)

    rename_map = {
        "timestamp": "Date",
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }

    df.rename(columns=rename_map, inplace=True)

    df.columns = [x.capitalize() for x in df.columns]

    df['Date'] = pd.to_datetime(df['Date'])

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    df = df.sort_values('Date', kind='mergesort')

    df.set_index('Date', inplace=True)

    # EMA
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA65'] = df['Close'].ewm(span=65, adjust=False).mean()

    # Volume Ratio
    df['Volume MA20'] = df['Volume'].rolling(20).mean()
    df['Volume Ratio'] = df['Volume'] / df['Volume MA20']

    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))

    df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['ATR14'] = df['TR'].rolling(14).mean()

    return df

# =========================
# BACKTEST
# =========================
def run_backtest(
    df,
    stock_name,
    nen_tich_luy,
    so_ngay_tich_luy,
    breakout_days_check,
    max_chase,
    target,
    stoploss,
    min_hold_days,
    max_hold_days,
    avoid_duplicate,
    min_volume_ratio,
    max_volume_ratio,
    use_atr,
    atr_multiplier
):

    trades = []
    last_breakout_idx = -1

    for i in range(80, len(df)):

        if i <= last_breakout_idx:
            continue

        # Trend Filter
        if df['EMA21'].iloc[i] <= df['EMA65'].iloc[i]:
            continue

        trade_found = False

        for base_days in so_ngay_tich_luy:

            if i + base_days >= len(df):
                continue

            base_df = df.iloc[i:i+base_days]

            highest = base_df['High'].max()
            lowest = base_df['Low'].min()

            base_range = (highest - lowest) / lowest

            if base_range > nen_tich_luy:
                continue

            breakout_start = i + base_days

            for breakout_idx in range(
                breakout_start,
                breakout_start + breakout_days_check
            ):

                if breakout_idx >= len(df):
                    continue

                candle = df.iloc[breakout_idx]

                breakout_price = candle['Close']

                # Breakout
                if breakout_price <= highest:
                    continue

                # Chase Filter
                if breakout_price > highest * max_chase:
                    continue

                # Volume Ratio Filter
                vr = candle['Volume Ratio']

                if vr < min_volume_ratio:
                    continue

                if vr > max_volume_ratio:
                    continue

                # TP/SL
                if use_atr:

                    atr = candle['ATR14']

                    tp = breakout_price + atr * atr_multiplier * 2
                    sl = breakout_price - atr * atr_multiplier

                else:

                    tp = breakout_price * target
                    sl = breakout_price * stoploss

                result = "HOLD"

                exit_price = None
                exit_date = None

                start_exit_idx = breakout_idx + min_hold_days

                end_exit_idx = min(
                    breakout_idx + max_hold_days,
                    len(df) - 1
                )

                future = df.iloc[start_exit_idx:end_exit_idx + 1]

                for j in range(len(future)):

                    row = future.iloc[j]

                    # Stoploss
                    if row['Low'] <= sl:

                        result = "STOPLOSS"
                        exit_price = sl
                        exit_date = future.index[j]

                        break

                    # Take Profit
                    if row['High'] >= tp:

                        result = "TAKE_PROFIT"
                        exit_price = tp
                        exit_date = future.index[j]

                        break

                # Force Exit
                if exit_price is None:

                    forced_exit_row = df.iloc[end_exit_idx]

                    exit_price = forced_exit_row['Close']
                    exit_date = forced_exit_row.name

                    result = "MAX_HOLD_EXIT"

                profit = (
                    exit_price - breakout_price
                ) / breakout_price

                trades.append({

                    'Buy Date': df.index[breakout_idx],
                    'Buy Price': breakout_price,
                    'Exit Date': exit_date,
                    'Exit Price': exit_price,
                    'Profit': profit,
                    'Result': result,
                    'Stock': stock_name,
                    'Volume Ratio': vr

                })

                last_breakout_idx = breakout_idx + avoid_duplicate

                trade_found = True

                break

            if trade_found:
                break

    return pd.DataFrame(trades)

# =========================
# STATISTICS
# =========================
def calculate_statistics(trades_df):

    if len(trades_df) == 0:
        return None

    total = len(trades_df)

    wins = trades_df[trades_df['Profit'] > 0]
    losses = trades_df[trades_df['Profit'] <= 0]

    winrate = len(wins) / total * 100

    avg_win = wins['Profit'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = abs(losses['Profit'].mean()) * 100 if len(losses) > 0 else 0

    gross_profit = wins['Profit'].sum()
    gross_loss = abs(losses['Profit'].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss != 0 else np.inf
    )

    expectancy = (
        (winrate / 100) * avg_win
        -
        (1 - winrate / 100) * avg_loss
    )

    return {
        'Total Trades': total,
        'Winrate %': round(winrate, 2),
        'Average Win %': round(avg_win, 2),
        'Average Loss %': round(avg_loss, 2),
        'Profit Factor': round(profit_factor, 2),
        'Expectancy %': round(expectancy, 2)
    }

# =========================
# MAIN UI
# =========================
st.title("📈 Crypto Spot Breakout Backtest")

if run_backtest_button and not uploaded_files:

    st.warning("Please upload CSV files")

elif run_backtest_button and uploaded_files:

    summary_results = []
    all_trades = []

    st.info(f"Processing {len(uploaded_files)} files...")

    for file in uploaded_files:

        try:

            stock_name = file.name.replace(".csv", "")

            df = read_crypto_file(file)

            trades_df = run_backtest(
                df,
                stock_name,
                **BACKTEST_CONFIG
            )

            trades_df = trades_df[
                (trades_df['Buy Date'] >= start_date)
                &
                (trades_df['Buy Date'] <= end_date)
            ]

            if len(trades_df) > 0:
                all_trades.append(trades_df)

            stats = calculate_statistics(trades_df)

            if stats is not None:

                stats['Coin'] = stock_name

                summary_results.append(stats)

        except Exception as e:

            st.error(f"Error {file.name}: {e}")

    # SUMMARY
    if len(summary_results) > 0:

        summary_df = pd.DataFrame(summary_results)

        summary_df = summary_df.sort_values(
            by='Expectancy %',
            ascending=False
        )

        st.header("🏆 FINAL SUMMARY")

        st.dataframe(summary_df, use_container_width=True)

    # ALL TRADES
    if len(all_trades) > 0:

        all_trades_df = pd.concat(all_trades, ignore_index=True)

        st.header("📜 ALL TRADES")

        st.dataframe(
            all_trades_df.sort_values(by='Buy Date'),
            use_container_width=True
        )

        # =========================
        # PORTFOLIO BACKTEST
        # =========================
        cash = INITIAL_CAPITAL

        open_positions = []

        executed_trades = []

        df_loop = all_trades_df.copy()

        df_loop = df_loop.sort_values(
            by='Buy Date',
            kind='mergesort'
        ).reset_index(drop=True)

        for idx, row in df_loop.iterrows():

            buy_date = row['Buy Date']

            # Close positions
            remaining_positions = []

            for pos in open_positions:

                if pos['Exit Date'] <= buy_date:

                    cash += pos['Exit Value']

                else:

                    remaining_positions.append(pos)

            open_positions = remaining_positions

            buy_price = row['Buy Price']

            # No lot size
            position_size = min(
                MAX_POSITION_SIZE,
                cash
            )

            if position_size <= 0:
                continue

            shares = position_size / buy_price

            # Buy Fee
            buy_fee = position_size * fee_rate

            total_buy_cost = position_size + buy_fee

            if cash < total_buy_cost:
                continue

            cash -= total_buy_cost

            gross_exit_value = shares * row['Exit Price']

            sell_fee = gross_exit_value * fee_rate

            exit_value = gross_exit_value - sell_fee

            pnl = exit_value - total_buy_cost

            open_positions.append({
                'Exit Date': row['Exit Date'],
                'Exit Value': exit_value
            })

            executed_trades.append({

                'Buy Date': row['Buy Date'],
                'Exit Date': row['Exit Date'],
                'Coin': row['Stock'],
                'Buy Price': buy_price,
                'Exit Price': row['Exit Price'],
                'Position Size': position_size,
                'PnL': pnl,
                'Profit %': pnl / position_size,
                'Cash After Buy': cash

            })

        # Close all remaining
        for pos in open_positions:
            cash += pos['Exit Value']

        if len(executed_trades) > 0:

            executed_df = pd.DataFrame(executed_trades)

            executed_df = executed_df.sort_values(
                by='Exit Date',
                kind='mergesort'
            )

            total_profit = cash - INITIAL_CAPITAL

            roi = total_profit / INITIAL_CAPITAL * 100

            # Equity Curve
            equity = INITIAL_CAPITAL

            equity_curve = []

            for _, r in executed_df.iterrows():

                equity += r['PnL']

                equity_curve.append(equity)

            executed_df['Equity'] = equity_curve

            executed_df['Peak'] = executed_df['Equity'].cummax()

            executed_df['Drawdown'] = (
                executed_df['Equity']
                -
                executed_df['Peak']
            ) / executed_df['Peak']

            max_drawdown = executed_df['Drawdown'].min()

            st.header("💰 PORTFOLIO RESULT")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Initial Capital",
                f"${INITIAL_CAPITAL:,.2f}"
            )

            c2.metric(
                "Final Capital",
                f"${cash:,.2f}"
            )

            c3.metric(
                "Total Profit",
                f"${total_profit:,.2f}"
            )

            c4.metric(
                "ROI %",
                f"{roi:.2f}%"
            )

            st.write(
                f"Executed Trades: {len(executed_df)}"
            )

            st.write(
                f"Max Drawdown: {max_drawdown:.2%}"
            )

            # Equity Curve
            st.subheader("📈 Equity Curve")

            fig_eq, ax_eq = plt.subplots(figsize=(14, 5))

            ax_eq.plot(
                executed_df['Exit Date'],
                executed_df['Equity']
            )

            ax_eq.grid(True)

            st.pyplot(fig_eq)

            plt.close(fig_eq)

            # Download
            csv_data = executed_df.to_csv(
                index=False
            ).encode('utf-8-sig')

            st.download_button(
                "📥 Download executed_trades.csv",
                csv_data,
                "executed_trades.csv",
                "text/csv"
            )
