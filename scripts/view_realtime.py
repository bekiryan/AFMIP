"""
View Real-time Stock Data (App.Alpaca.Markets)
─────────────────────────────────────────────
Simple utility to fetch live quotes or snapshots for a given list of tickers
using the Alpaca API. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env.

Usage
─────
    # Fetch live snapshot for SPY, AAPL, MSFT
    python scripts/view_realtime.py --tickers SPY,AAPL,MSFT

    # Fetch live quotes only
    python scripts/view_realtime.py --tickers TSLA,NVDA --quotes-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.loader import load_realtime_quotes, load_realtime_snapshots

console = Console()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View live stock data from Alpaca")
    p.add_argument("--tickers", type=str, default="SPY,AAPL,MSFT",
                   help="Comma-separated list of tickers (default: SPY,AAPL,MSFT)")
    p.add_argument("--quotes-only", action="store_true",
                   help="Fetch quotes instead of full snapshot")
    return p.parse_args()

def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    console.rule("[bold blue]Alpaca Real-time Data")

    try:
        if args.quotes_only:
            console.print(f"Fetching live quotes for: [cyan]{', '.join(tickers)}[/cyan]")
            df = load_realtime_quotes(tickers)
        else:
            console.print(f"Fetching live snapshots for: [cyan]{', '.join(tickers)}[/cyan]")
            df = load_realtime_snapshots(tickers)

        if df.empty:
            console.print("[yellow]No data returned from Alpaca.[/yellow]")
            return

        # Display as a rich table
        table = Table(show_header=True, header_style="bold magenta")
        
        # Add columns
        for col in df.columns:
            table.add_column(str(col))

        # Add rows
        for _, row in df.iterrows():
            table.add_row(*[str(val) for val in row])
        
        console.print(table)

    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]API Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
