# QUERY

This module searches for TEQ subgraphs that match particular structural/label/attribute parameters.

## Outline

In traditional structured query language (SQL), a query at least consists of a `return-fields` and `condition`. E.g.:

```
SELECT username, password
FROM account JOIN violation ON account.id=violation.account_id
WHERE violation.level='CRITICAL';
```

return-fields are `username` and `password` of the account model.

conditions are that the accounts have violations of a CRITICAL `level`.

TEQ's graph querying module uses a json object to represent `condition`.

The json object encodes expected matching subgraph's structure and each operator's attribute.

The condition object has the following schema (protobuf):

```
message Node {
    oneof node {
        double cst = 1;
        Variable var = 2;
        Operator op = 3;
    }
}

message Variable {
    string label = 1;
    string dtype = 2;
    repeated uint32 shape = 3;
}

message Operator {
    string opname = 1;
    map<string,Attribute> attrs = 2;
    repeated Node args = 3;
}

message Attribute {
    oneof val {
        int inum = 1;
        double dnum = 2;
        repeated int inums = 3;
        repeated double dnums = 4;
        string str = 5;
        Node node = 6;
        Layer layer = 7;
    }
}

message Layer {
    string name = 1;
    Node input = 2;
}
```

## Example

Assuming we have the following TEQ graph

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

The following parameters:

```
condition = `{
    "opname": "ADD",
    "args": [
        {},
        {
            "opname": "MATMUL"
        }
    ]
}`
depth = 2
```

yields:

```
[
    {
        "opname": "ADD",
        "attribute": ...,
        "args": [
            {
                "opname": "EXP",
                "attribute": ...,
                "args": [
                    {
                        "label": "B",
                        "shape": [2, 2],
                    }
                ]
            },
            {
                "opname": "MATMUL",
                "attribute": ...,
                "args": [
                    {
                        "opname": "NEG"
                        "attribute": ...
                    },
                    5.0
                ]
            }
        ]
    }
]
```

The following parameter:

```
condition = `{
    "opname": "ADD"
}`
depth = 1
```

yields:

```
[
    {
        "opname": "ADD":
        "attribute": ...,
        "args": [
            {
                "opname": "EXP"
                "attribute": ...
            },
            {
                "opname": "MATMUL"
                "attribute": ...
            }
        ]
    },
    {
        "opname": "ADD":
        "attribute": ...,
        "args": [
            {
                "opname": "SIGMOID"
                "attribute": ...
            },
            {
                "label": "A",
                "shape": [3, 2]
            }
        ]
    }
]
```