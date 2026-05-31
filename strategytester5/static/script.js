
// -----------------------------------
// PLOTLY CHART
// -----------------------------------

const balanceTrace = {

    x: [],
    y: [],

    mode: "lines",
    name: "Balance",

    line: {
        color: "dodgerblue",
        width: 3
    },

    hoverlabel: {

        bgcolor: "#08111f",
        bordercolor: "dodgerblue",
        font: {
            color: "white",
            family: "Ubuntu"
        }
    },

    hovertemplate:
        "<b>Balance</b><br>" +
        "Time: %{x}<br>" +
        "Value: %{y:.2f}<extra></extra>",
};

const equityTrace = {

    x: [],
    y: [],

    mode: "lines",
    name: "Equity",

    line: {
        color: "#4dff88",
        width: 3
    },

    hoverlabel: {

        bgcolor: "#08111f",
        bordercolor: "dodgerblue",
        font: {
            color: "white",
            family: "Ubuntu"
        }
    },

    hovertemplate:
        "<b>Equity</b><br>" +
        "Time: %{x}<br>" +
        "Value: %{y:.2f}<extra></extra>",
};

const layout = {

    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",

    font: {
        color: "white",
        family: "Ubuntu",
        size: 14
    },

    xaxis: {

        title: "Time",

        gridcolor: "rgba(255,255,255,0.05)"
    },

    yaxis: {

        title: "Account",
        gridcolor: "rgba(255,255,255,0.05)"
    },

    legend: {

        orientation: "h",

        y: 1.1
    },

    margin: {
        l: 50,
        r: 20,
        t: 30,
        b: 50
    },

    hovermode: "x unified",

    hoverlabel: {
        bgcolor: "#08111f",
        bordercolor: "dodgerblue",
        font: {
            color: "white",
            family: "Ubuntu",
            size: 14
        }
    },
};

Plotly.newPlot(
    "equityChart",
    [balanceTrace, equityTrace],
    layout,
    {
        responsive: true
    }
);

function formatMoney(value) {

    return "$" + Number(value).toLocaleString(
        undefined,
        {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }
    );
}

function formatDateTime(timestamp)
{
    const d = new Date(timestamp * 1000);

    const year = d.getFullYear();

    const month =
        String(d.getMonth() + 1)
            .padStart(2, "0");

    const day =
        String(d.getDate())
            .padStart(2, "0");

    const hour =
        String(d.getHours())
            .padStart(2, "0");

    const minute =
        String(d.getMinutes())
            .padStart(2, "0");

    const second =
        String(d.getSeconds())
            .padStart(2, "0");

    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

// --------------------------------------------------
// UPDATE DASHBOARD
// --------------------------------------------------

function updateDashboard(data) {

    // -------------------------
    // ACCOUNT INFO
    // -------------------------

    document.getElementById("balance").innerText =
        formatMoney(data.balance);

    document.getElementById("equity").innerText =
        formatMoney(data.equity);

    document.getElementById("free_margin").innerText =
        formatMoney(data.free_margin);

    // -------------------------
    // EQUITY CURVE
    // -------------------------

    const now =
        new Date(data.time * 1000);

    Plotly.extendTraces(

        "equityChart",

        {
            x: [[now], [now]],

            y: [
                [data.balance],
                [data.equity]
            ]
        },

        [0, 1],
    );

    // -------------------------
    // TRADES TABLE
    // -------------------------

    const tbody = document.getElementById("trades-body");

    tbody.innerHTML = "";

    data.trades.forEach(trade => {

        const isPending =
            trade.type > 1;

        const rowStyle =
            isPending
            ? 'style="background: rgba(255,255,255,0.04);"'
            : '';

        const profitClass =
            trade.profit >= 0
            ? "profit-positive"
            : "profit-negative";

        tbody.innerHTML += `

            <tr ${rowStyle}>

                <td>${trade.symbol}</td>

                <td>${trade.ticket}</td>

                <td>
                    ${
                        trade.time
                        ? formatDateTime(trade.time)
                        : "-"
                    }
                </td>

                <td>${trade.type}</td>

                <td>
                    ${
                        trade.volume ??
                        trade.volume_current ??
                        "-"
                    }
                </td>

                <td>
                    ${
                        trade.price_open ??
                        "-"
                    }
                </td>

                <td>${trade.sl ?? "-"}</td>

                <td>${trade.tp ?? "-"}</td>

                <td>
                    ${
                        trade.price_current ??
                        "-"
                    }
                </td>

                <td class="${profitClass}">
                    ${
                        trade.profit != null
                        ? trade.profit.toFixed(2)
                        : "-"
                    }
                </td>

            </tr>

        `;
    });
}

function renderEntriesPLPlot(plotJson)
{
    const section =
        document.getElementById(
            "entries-pl-section"
        );

    section.style.display =
        "block";

    section.innerHTML = `
        <div
            id="entries-pl-container"
            style="
                width:100%;
                min-height:700px;
            ">
        </div>
    `;

    const fig =
        JSON.parse(plotJson);

    Plotly.newPlot(
        "entries-pl-container",
        fig.data,
        fig.layout,
        {
            responsive: true
        }
    );

    window.dispatchEvent(
        new Event("resize")
    );
}

function renderReport(data)
{
    console.log("REPORT DATA:", data);

    const liveSection =
        document.getElementById(
            "live-trades-section"
        );

    liveSection.style.display = "none";

    // backtest report section

    const backtest_report_section = document.getElementById("backtest-report-section");
    backtest_report_section.style.display = "block";
    
    tester_stats = data.tester_stats

    backtest_report_section.innerHTML = `

        <table class="report-table">
            <tbody>
                <tr>
                    <th>Initial Deposit</th>
                    <td class="number">${tester_stats.initial_deposit}</td>

                    <th>Ticks</th>
                    <td class="number">${tester_stats.ticks}</td>

                    <th>Symbols</th>
                    <td class="number">${tester_stats.symbols}</td>
                </tr>

                <tr>
                    <th>Total Net Profit</th>
                    <td class="number">${tester_stats.net_profit.toFixed(2)}</td>

                    <th>Balance Drawdown Absolute</th>
                    <td class="number">${tester_stats.balance_drawdown_absolute.toFixed(2)}</td>

                    <th>Equity Drawdown Absolute</th>
                    <td class="number">${tester_stats.equity_drawdown_absolute.toFixed(2)}</td>
                </tr>

                <tr>
                    <th>Gross Profit</th>
                    <td class="number">${tester_stats.gross_profit.toFixed(2)}</td>

                    <th>Balance Drawdown Maximal</th>
                    <td class="number">
                        ${tester_stats.balance_drawdown_maximal.toFixed(2)}
                        (${(tester_stats.balance_drawdown_maximal / 100).toFixed(2)}%)
                    </td>

                    <th>Equity Drawdown Maximal</th>
                    <td class="number">
                        ${tester_stats.equity_drawdown_maximal.toFixed(2)}
                        (${(tester_stats.equity_drawdown_maximal / 100).toFixed(2)}%)
                    </td>
                </tr>

                <tr>
                    <th>Gross Loss</th>
                    <td class="number">${tester_stats.gross_loss.toFixed(2)}</td>

                    <th>Balance Drawdown Relative</th>
                    <td class="number">
                        ${tester_stats.balance_drawdown_relative.toFixed(2)}%
                        (${(tester_stats.balance_drawdown_relative * 100).toFixed(2)})
                    </td>

                    <th>Equity Drawdown Relative</th>
                    <td class="number">
                        ${tester_stats.equity_drawdown_relative.toFixed(2)}%
                        (${(tester_stats.equity_drawdown_relative * 100).toFixed(2)})
                    </td>
                </tr>

                <tr>
                    <th>Profit Factor</th>
                    <td class="number">${tester_stats.profit_factor.toFixed(2)}</td>

                    <th>Expected Payoff</th>
                    <td class="number">${tester_stats.expected_payoff.toFixed(2)}</td>

                    <th>Margin Level</th>
                    <td class="number">${tester_stats.margin_level.toFixed(2)}%</td>
                </tr>

                <tr>
                    <th>Recovery Factor</th>
                    <td class="number">${tester_stats.recovery_factor.toFixed(2)}</td>

                    <th>Sharpe Ratio</th>
                    <td class="number">${tester_stats.sharpe_ratio.toFixed(2)}</td>

                    <th>Z-Score</th>
                    <td class="number">${tester_stats.z_score.toFixed(2)}</td>
                </tr>

                <tr>
                    <th>AHPR</th>
                    <td class="number">
                        ${tester_stats.ahpr_factor.toFixed(4)}
                        (${tester_stats.ahpr_percent.toFixed(2)}%)
                    </td>

                    <th>LR Correlation</th>
                    <td class="number">${tester_stats.lr_correlation.toFixed(2)}</td>

                    <th>OnTester Result</th>
                    <td class="number">${tester_stats.on_tester_results}</td>
                </tr>

                <tr>
                    <th>GHPR</th>
                    <td class="number">
                        ${tester_stats.ghpr_factor.toFixed(4)}
                        (${tester_stats.ghpr_percent.toFixed(2)}%)
                    </td>

                    <th>LR Standard Error</th>
                    <td class="number">${tester_stats.lr_standard_error.toFixed(2)}</td>

                    <td></td>
                    <td></td>
                </tr>

                <tr>
                    <th>Total Trades</th>
                    <td class="number">${tester_stats.total_trades}</td>

                    <th>Short Trades (won %)</th>
                    <td class="number">
                        ${tester_stats.short_trades_won}
                        (${(
                            100 *
                            tester_stats.short_trades_won /
                            Math.max(tester_stats.total_short_trades, 1)
                        ).toFixed(2)}%)
                    </td>

                    <th>Long Trades (won %)</th>
                    <td class="number">
                        ${tester_stats.long_trades_won}
                        (${(
                            100 *
                            tester_stats.long_trades_won /
                            Math.max(tester_stats.total_long_trades, 1)
                        ).toFixed(2)}%)
                    </td>
                </tr>

                <tr>
                    <th>Total Deals</th>
                    <td class="number">${tester_stats.total_deals}</td>

                    <th>Profit Trades (% of total)</th>
                    <td class="number">
                        ${tester_stats.profit_trades}
                        (${(
                            100 *
                            tester_stats.profit_trades /
                            Math.max(tester_stats.total_trades, 1)
                        ).toFixed(2)}%)
                    </td>

                    <th>Loss Trades (% of total)</th>
                    <td class="number">
                        ${tester_stats.loss_trades}
                        (${(
                            100 *
                            tester_stats.loss_trades /
                            Math.max(tester_stats.total_trades, 1)
                        ).toFixed(2)}%)
                    </td>
                </tr>

                <tr>
                    <th>Largest Profit Trade</th>
                    <td class="number">${tester_stats.largest_profit_trade.toFixed(2)}</td>

                    <th>Largest Loss Trade</th>
                    <td class="number">${tester_stats.largest_loss_trade.toFixed(2)}</td>

                    <td></td>
                    <td></td>
                </tr>

                <tr>
                    <th>Average Profit Trade</th>
                    <td class="number">${tester_stats.average_profit_trade.toFixed(2)}</td>

                    <th>Average Loss Trade</th>
                    <td class="number">${tester_stats.average_loss_trade.toFixed(2)}</td>

                    <td></td>
                    <td></td>
                </tr>

                <tr>
                    <th>Max Consecutive Wins ($)</th>
                    <td class="number">
                        ${tester_stats.maximum_consecutive_wins_count}
                        (${tester_stats.maximum_consecutive_wins_money.toFixed(2)})
                    </td>

                    <th>Max Consecutive Losses ($)</th>
                    <td class="number">
                        ${tester_stats.maximum_consecutive_losses_count}
                        (${tester_stats.maximum_consecutive_losses_money.toFixed(2)})
                    </td>

                    <td></td>
                    <td></td>
                </tr>

                <tr>
                    <th>Maximal Consecutive Profit (count)</th>
                    <td class="number">
                        ${tester_stats.maximal_consecutive_profit_count}
                        (${tester_stats.maximal_consecutive_profit_money.toFixed(2)})
                    </td>

                    <th>Maximal Consecutive Loss (count)</th>
                    <td class="number">
                        ${tester_stats.maximal_consecutive_loss_count}
                        (${tester_stats.maximal_consecutive_loss_money.toFixed(2)})
                    </td>

                    <td></td>
                    <td></td>
                </tr>

                <tr>
                    <th>Average Consecutive Wins</th>
                    <td class="number">${tester_stats.average_consecutive_wins.toFixed(2)}</td>

                    <th>Average Consecutive Losses</th>
                    <td class="number">${tester_stats.average_consecutive_losses.toFixed(2)}</td>

                    <td></td>
                    <td></td>
                </tr>
            </tbody>
        </table>
    `;

    // Holding time plots

    renderEntriesPLPlot(data.entries_plot)

    const holdingSection = document.getElementById("holding-section");
    holdingSection.style.display = "block";

    const holding_stats = data.holding_stats;
    const holding_plot = data.holding_plot

    holdingSection.innerHTML = `
        <div class="holding-grid">

            <div
                id="holding-plot-container"
                class="holding-chart">
            </div>

            <div class="holding-stats">
                <table class="report-table">
                    <tbody>
                        <tr>
                            <th>Mean</th>
                            <td>${holding_stats.mean}</td>
                        </tr>

                        <tr>
                            <th>Std</th>
                            <td>${holding_stats.std}</td>
                        </tr>

                        <tr>
                            <th>Min</th>
                            <td>${holding_stats.min}</td>
                        </tr>

                        <tr>
                            <th>25%</th>
                            <td>${holding_stats.q25}</td>
                        </tr>

                        <tr>
                            <th>Median</th>
                            <td>${holding_stats.median}</td>
                        </tr>

                        <tr>
                            <th>75%</th>
                            <td>${holding_stats.q75}</td>
                        </tr>

                        <tr>
                            <th>Max</th>
                            <td>${holding_stats.max}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // container exists

    const fig =
        JSON.parse(
            data.holding_plot
        );

    Plotly.newPlot(
        "holding-plot-container",
        fig.data,
        fig.layout,
        {
            responsive: true
        }
    );
}

// ------------------ Socket listeners ---------------------

const socket = io();

socket.on(
    "connect",
    () => {
        console.log(
            "Socket connected:",
            socket.id
        );
    }
);


socket.on(
    "dashboard_update",
    function(data){
        updateDashboard(data)
    }
);


// listen for the finish in simulation (backtesting)

socket.on(
    "simulation_finished",
    function(data){
        renderReport(data)
    }
);


socket.on(
    "disconnect",
    () => {
        console.log(
            "Socket disconnected:",
            socket.id
        );
    }
);