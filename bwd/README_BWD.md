# Backwards (BWD)

Backwards implements a Grader ADE graph traveler to generate partial derivatives using Automatic Differentiation (AD).

Every Grader requires an instance of iRuleSet.
By default, Grader will use assign a global default_rules if user doesn't provide one on construction.
However the default_rules is not initialized in this library, so the user must initialize the Grader::defaut_rules before using Grader.
