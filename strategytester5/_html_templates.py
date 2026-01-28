def html_report_template() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Strategy Tester Report</title>

    <!-- Bootstrap 5 -->
    <link
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        rel="stylesheet"
    >

    <style>
        :root {
            --bg: #ffffff;
            --border: #dee2e6;
            --header-bg: #f3f4f6;
            --text: #212529;
            --muted: #6c757d;
        }

        body {
            background: var(--bg);
            font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
            font-size: 11px;
            color: var(--text);
            line-height: 1.35;
        }

        /* Page width */
        .page-wrapper {
            margin: 24px 10%;
        }

        h4 {
            font-size: 13px;
            font-weight: 600;
            text-align: center;
            margin: 22px 0 10px;
        }

        .muted {
            color: var(--muted);
            font-size: 10px;
        }

        /* Sections are logical only – no visual boxes */
        .section {
            margin-bottom: 18px;
        }

        /* Unified report table style */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
        }

        th, td {
            padding: 4px 6px;
            border: 1px solid var(--border);
            text-align: center;
            vertical-align: middle;
            white-space: nowrap;
        }

        th {
            background: var(--header-bg);
            font-weight: 600;
        }

        td.number {
            text-align: right;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* Stats table slightly larger */
        .report-table th,
        .report-table td {
            font-size: 11px;
            padding: 6px 8px;
        }

        /* Curve */
        .curve-wrapper {
            display: flex;
            justify-content: center;
            margin: 12px 0 16px;
        }

        .curve-img {
            max-width: 60%;
        }

        /* Clean Bootstrap interference */
        .table-striped > tbody > tr:nth-of-type(odd) {
            background-color: #fafafa;
        }

        .table-responsive {
            overflow-x: auto;
        }
        
        /* ---------- Responsive Scaling ---------- */

        /* Page width adapts */
        .page-wrapper {
            margin: clamp(12px, 20vw, 20%);
        }

        /* Global font scales with viewport */
        body {
            font-size: clamp(9px, 0.75vw, 11px);
        }

        /* Headers scale gently */
        h4 {
            font-size: clamp(11px, 1vw, 13px);
        }

        /* Table font & spacing scale */
        table {
            font-size: clamp(8px, 0.7vw, 10px);
        }

        th, td {
            padding: clamp(2px, 0.4vw, 6px);
        }

        /* Stats table slightly larger */
        .report-table th,
        .report-table td {
            font-size: clamp(9px, 0.8vw, 11px);
        }

        /* Curve responsiveness */
        .curve-img {
            max-width: 100%;
            width: clamp(280px, 60vw, 900px);
            height: auto;
        }

        /* Allow horizontal scroll ONLY if needed */
        .table-responsive {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        /* Reduce whitespace on small screens */
        @media (max-width: 900px) {
            h4 {
                margin: 16px 0 8px;
            }

            .section {
                margin-bottom: 12px;
            }
        }

        /* Extreme small screens (last resort) */
        @media (max-width: 600px) {
            .page-wrapper {
                margin: 8px;
            }

            table {
                font-size: 8px;
            }

            th, td {
                padding: 2px 3px;
            }
        }

    </style>

</head>
<body>

<div class="page-wrapper">

    <h4>Strategy Tester Report<br><span class="muted">Python Simulator</span></h4>

    <h4>Stats</h4>
    <div class="section">
        {{STATS_TABLE}}
    </div>

    <div class="section">
        <div class="curve-wrapper">
            {{CURVE_IMAGE}}
        </div>
    </div>

    <h4>Orders</h4>
    <div class="section table-responsive">
        <table class="table table-sm table-striped align-middle">
            <thead>
                <tr>
                    <th>Open Time</th>
                    <th>Order</th>
                    <th>Symbol</th>
                    <th>Type</th>
                    <th class="number">Volume</th>
                    <th class="number">Price</th>
                    <th class="number">S / L</th>
                    <th class="number">T / P</th>
                    <th>Time</th>
                    <th>State</th>
                    <th>Comment</th>
                </tr>
            </thead>
            <tbody>
                {{ORDER_ROWS}}
            </tbody>
        </table>
    </div>

    <h4>Deals</h4>
    <div class="section table-responsive">
        <table class="table table-sm table-striped align-middle">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Deal</th>
                    <th>Symbol</th>
                    <th>Type</th>
                    <th>Entry</th>
                    <th class="number">Volume</th>
                    <th class="number">Price</th>
                    <th class="number">Commission</th>
                    <th class="number">Swap</th>
                    <th class="number">Profit</th>
                    <th>Comment</th>
                    <th class="number">Balance</th>
                </tr>
            </thead>
            <tbody>
                {{DEAL_ROWS}}
            </tbody>
        </table>
    </div>

</div>

</body>
</html>

"""