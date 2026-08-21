# Contributing

1. Keep datasets, checkpoints, caches, environments, and generated evaluations
   out of Git.
2. Preserve the method-only boundary documented in `README.md`.
3. Run `python -m unittest discover -s tests` in the documented environment.
4. Keep published dataset input-contract changes explicit, versioned, and
   fail-fast when a schema is unsupported.
5. Never commit credentials, private keys, subscription URLs, or machine-local
   absolute paths.

Bug reports should include the command, configuration, relevant manifest IDs,
and the smallest reproducible example. Do not attach restricted datasets or
model checkpoints.
