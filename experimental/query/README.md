# Implicit GQL Schema

Query accepts GQL commands assuming we have the following implicit schema:

```
union Node = Data | Operator;

type Layer {
    name: String!
    input: Node!
}

union Attribute = Node | Layer | String | Int | Float | [Int] | [Float]

type Data implements Node {
    shape: [Int!]
    dtype: String!
    label: String
    immutable: Boolean
}

type Operator implements Node {
    opname: String!
    children: [Node!]!
    attribute: [Attribute]
}
```

For Query interface, accepted commands rejects mutations.
