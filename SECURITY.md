# Security

## Weights

ASTRAUTOMA loads AI weights with `torch.load(weights_only=True)`, so a weight file is read only as
numbers and cannot run code. Still, only install weights from the
[official releases](https://github.com/himsson/ASTRAUTOMA/releases) or files you trust.

## Network

The app talks only to the kRPC server of your own game (by default `127.0.0.1`). It does not send
data anywhere else.

## Reporting a problem

If you find a security problem, **do not open a public issue**. Report it privately through
[Security → Report a vulnerability](https://github.com/himsson/ASTRAUTOMA/security/advisories/new).
You will get an answer within a week.
