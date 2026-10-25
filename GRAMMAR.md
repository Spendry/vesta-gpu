# Grammar

Four forms. Everything else is naming.

## The forms

| Form | Says | Written |
|---|---|---|
| **law** | a value follows from others, the same rule at every index | `a + b`, `a * b`, `exp(a)`, `a[i*64 + j]` |
| **fold** | many values become one, along one axis | `fold sum i: e`, `fold max i: e`, `fold min i: e`, `fold step i: init as c ... end` |
| **fork** | a choice made on a live value | `fork c ? a : b` |
| **residue** | a value that cannot be derived, so it crosses in or out | `residue w[i, j]` (receive), `residue w[i, j] <- e` (emit) |

`fold step` is a fold whose combine is a block instead of `+` or `max`.
The carried value is `c`. The block ends with `-> e`, the next `c`.

## Statements

```
axis NAME = e                    # an index range 0 .. e-1. e may use residue scalars
residue NAME[AXES]               # receive
residue NAME[e, ...] <- e        # emit
NAME[AXES] = e                   # name a value
NAME[AXES] = fold step AX: e as NAME
    statements
    -> e
end
```

`[AXES]` is left out for a single value. `#` starts a comment.

## Expressions

```
e := fork c ? e : e | fold OP AXIS: e | c
c := s (== | < | > | <= | >=) s | s
s := m ((+ | -) m)*
m := u ((* | /) u)*
u := - u | p
p := atom [ '[' e, ... ']' ]
atom := NUMBER | NAME | FUNC '(' e ')' | '(' e ')'
OP := sum | max | min        FUNC := exp | tanh | sqrt
```

A fold or fork runs to the end of its expression. Wrap it in brackets to use it inside one.

## Meaning

- An axis name used as a value is its index, 0 to its extent minus 1.
- A value carries the axes it uses. Operands line up by axis name, not by position.
- `fold OP i:` removes axis `i`.
- Indexing takes any law of axes, for example `qkv[768 + h*64 + e]`.
- An assignment may only use axes named on its left side.
- Statements run in order. An emit is seen by every read after it.
