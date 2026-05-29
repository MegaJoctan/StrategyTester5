
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

        title: "Account Value",
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

// --------------------------------------------------
// UPDATE DASHBOARD
// --------------------------------------------------

async function updateDashboard() {

    const response =
        await fetch("/dashboard");

    const data =
        await response.json();

    // -----------------------------------
    // ACCOUNT INFO
    // -----------------------------------

    document.getElementById("balance").innerText = formatMoney(live_data.balance);
    document.getElementById("equity").innerText = formatMoney(live_data.equity);
    document.getElementById("free_margin").innerText = formatMoney(live_data.free_margin);

    // -----------------------------------
    // UPDATE PLOTLY
    // -----------------------------------

    const now = new Date(data.time * 1000);

    Plotly.extendTraces(

        "equityChart",

        {
            x: [[now], [now]],

            y: [
                [live_data.balance],
                [live_data.equity]
            ]
        },

        [0, 1],

        200 // max points
    );

    // -----------------------------------
    // TRADES TABLE
    // -----------------------------------

    const tbody =
        document.getElementById("trades-body");

    tbody.innerHTML = "";

    live_data.trades.forEach(trade => {

        // pending order if type > 1
        const isPending =
            trade.type > 1;

        // darker row
        const rowStyle =
            isPending
            ? 'style="background: rgba(255,255,255,0.04);"'
            : '';

        // profit colors
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
                        ? new Date(trade.time * 1000)
                            .toLocaleTimeString()
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

// ---------------------------------------
// REFRESH LOOP
// ---------------------------------------

setInterval(updateDashboard, 1000);

updateDashboard();
