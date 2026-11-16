# Security

Weights are loaded with `torch.load(weights_only=True)`, so a weights file can't run code. Still,
only install weights from the releases here.

The app only talks to the kRPC server of your own game (127.0.0.1 by default) and sends nothing
anywhere else.

If you find a security problem please report it privately through
[Security > Report a vulnerability](https://github.com/himsson/ASTRAUTOMA/security/advisories/new)
and not in a public issue.
