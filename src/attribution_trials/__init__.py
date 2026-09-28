"""Attribution Trials: auditing whether a user simulator uses the target user's information.

Each audited simulator is scored on a user's held-out behavior under five
conditions: no user information, the population, the user's group, a matched
imposter from the same group, and the target user. The gains between
conditions show whether the target's own information helps beyond what the
controls already provide.
"""

__version__ = "0.1.0"
