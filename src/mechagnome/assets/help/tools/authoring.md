# Authoring tools

Tool source must define one synchronous entry point:

```python
def main(input, ctx):
    return {"result": input["value"] * 2}
```

`input` is a dictionary. The return value must be JSON-serializable. Imports,
filesystem access, and subprocesses are ordinary Python capabilities.

This is trusted code execution, not a security sandbox. Run Mechagnome only in
a disposable, tightly isolated environment.
