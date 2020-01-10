# Implicit GQL Schema

Query accepts GQL commands assuming we have the following implicit schema:

```
type Layer {
    name: String!
    input: Node!
}

union AttrVal = Node | Layer | String | Int | Float | [Int] | [Float]

type Attribute {
    key: String
    val: AttrVal
}

# include constant scalar
union Node = Constant | Variable | Operator

type Variable {
    name: String
    dtype: String!
    shape: [Int!]
}

type Operator {
    name: String
    args: [Node]
    attrs: Attribute
}

type Constant {
    scalar: Float
}
```

For Query interface, accepted commands rejects mutations.

assuming we have the following TEQ graph

```
tenncor::EVariable var = eteq::make_variable_scalar<int32_t>(3, teq::Shape({3, 2}), "A");
tenncor::EVariable var2 = eteq::make_variable_scalar<int32_t>(4, teq::Shape({2, 2}), "B");
tenncor::ETensor cst = eteq::make_constant_scalar<int32_t>(5, teq::Shape({2, 3}));

auto root = tenncor::sigmoid(tenncor::exp(var2) + tenncor::matmul(-var, cst)) + var;

// ADD
// `-- SIGMOID
// |   `-- ADD
// |       `-- EXP
// |       |   `-- B
// |       `-- MATMUL
// |           `-- NEG
// |           |   `-- A
// |           `-- 5
// `-- A
```

sample query:

```
query SelectPath($root: Node, $depth int) {
    ... on Constant {
        scalar
    }
    ... on Variable {
        name
        shape
    }
    ... on Operator {
        name
        args
    }
}
```

The following sample data 1:

```
SelectPath({
    "name": "ADD",
    "args": [
        {},
        {
            "name": "MATMUL"
        }
    ]
}, 2)
```

yields:

```
[
    {
        "name": "ADD",
        "args": [
            {
                "name": "EXP",
                "args": [
                    {
                        "name": "B",
                        "shape": [2, 2],
                    }
                ]
            },
            {
                "name": "MATMUL",
                "args": [
                    {
                        "name": "NEG"
                    },
                    {
                        "scalar": 5.0
                    }
                ]
            }
        ]
    }
]
```

The following sample data 2:

```
SelectPath({
    "name": "ADD"
}, 1)
```

yields:

```
[
    {
        "name": "ADD":
        "args": [
            {
                "name": "EXP"
            },
            {
                "name": "MATMUL"
            }
        ]
    },
    {
        "name": "ADD":
        "args": [
            {
                "name": "SIGMOID"
            },
            {
                "name": "A",
                "shape": [3, 2]
            }
        ]
    }
]
```
