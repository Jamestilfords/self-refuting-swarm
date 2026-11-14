# Run artifacts

The planned public experiment record follows slide 9:

- `ledger.jsonl`: validation results and execution status per proposal and cell.
- `decisions/<rule>/ord_<number>.jsonl`: acceptance decisions per ordering.
- `DECISIONS.sha256`: hash created when the decision logs are frozen.
- `exp2/`: Experiment 2 run records.

These files are created by future implementations, not by this scaffold.
Commit the decision logs and their hash before unblinding. Test arrays and
test scores live in a separate sealed volume, never in these logs.
