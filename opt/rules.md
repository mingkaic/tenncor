# **Optimization Rules Explained in Natural Language**

## **Module Purpose**
---

The optimization module takes TEQ graphs as an input and outputs TEQ graphs whereby specified input roots remain unchanged.

The module matches input TEQ graph against **matchers**.

Each matcher is associated with a corresponding **target**.

For *matched* subgraphs of input TEQ graph, the module converts that subgraph with associated *target*.

CversionCtx are matcher-target pairs that make up the basic components of the optimization module.

The .rules minilanguage defines conversions.

# Minilanguage .rule
---

In every .rules file, conversions are separated by ';'. Newlines and spaces are ignored.

Precedence of conversions are by order of statements defined.
That is, the first conversion specified will be applied before the second, and so on.

## **Keywords**

comm - declares that the following symbol is a commutative function

## **Primitives**

- number: a double precision decimal in base-10 form without scientific notation (hex and binary not supported)
- symbol: an alphabetic word with the following constraints:
    - no longer than 31 letters in length (symbols longer than 31 are unrecognized)
    - may contain underscores ('_')

## **Conversion Syntax**

A conversion statement has the following syntax:
```
Matcher => Target
```

## Matcher Syntax
---

A **matcher** is a function (commutative or non-commutative)

The function takes the form:

```
[ <keyword> ] <function name> [ <attribute> ] '(' <argument> [ '=' <attribute> ] [ ',' <more...> ] [ ',' '..' <variadics> ] ')'
```

where symbols wrapped in:
- `<>` denote variables
- `''` denote literals
- `[]` denote optional parts

The `<keyword>` can be `comm` which specifies how the arguments are matched:
- `comm` arguments are matched in any order.
- without this keyword, arguments are matched in specified order.

The `<function name>` has the same constraint as a symbol specified above.

An `<argument>` can be a number, symbol, or function

An `<variadics>` is a symbol denoting all the remaining arguments of the function.
Since variadics specify multiple arguments, they cannot specify attributes.

An `<attribute>` is a map in the form:

```
'{' <key> ':' <value> [ ',' <more key-value...> ] '}'
```

The attribute's `<key>` is a symbol, and `<value>` can be a number or array of number defined as:

```
'[' <number> [ ',' <other number...> ] ']'
```

### Note on Attribute

Specifying attribute adds a matcher constraint whereby the `marsh::Maps` returned from `iEdge::get_attrs` must contains a key-value pair equal to specified key-value pair.

### Note on Variadic Arguments and Commutative functions

Due .rule supporting both variadic and commutative functions, it is impractical for commutative AND variadic functions to hold function argument of height greater than one.

If a function is a commutative AND contains a variadic argument, it cannot hold more than one function argument, and that function argument cannot contain variadic arguments or function arguments.

Additionally the matcher cannot have more than one commutative function.

The following are disallowed:

- comm Foo(comm Bar(X))
- Foo(comm Bar(X), comm Baz(X))
- comm Foo(Bar(Baz(X)),..Y)
- comm Foo(Baz(X),Bar(Y),..Y)
- comm Foo(Bar(X,..Y),..Z)

## Target Syntax
---

A **target** can be any of the following:

- a number constant
- a symbol
- a function (without any keywords before the function name)

### **Target Function Syntax**

A target function is much like matcher function except it takes the form:

```
<function name> [ <attribute> ] '(' <argument> [ '=' <attribute> ] [ ',' <more...> ] [ ',' '..' <variadics> ] ')'
```

or

```
<function name> [ <attribute> ] '(' '..' <variadics> ')'
```

Since commutativity specifies argument order when matching, targets don't care about this property.

Target also allows functions to contain only the variadic arguments.
This is used to reduce the number of arguments (a common optimization method).

## Comments

Single line comment are double slashes "//".
