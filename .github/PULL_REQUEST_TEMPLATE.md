## What does this PR do?

<!-- One paragraph describing the change and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor (no behavior change)
- [ ] Documentation
- [ ] CI / tooling

## Related issues

Closes #<!-- issue number -->

## Testing

<!-- How did you test this? What commands did you run? -->

```bash
# C++ tests
ctest --test-dir build --output-on-failure

# Go tests
cd go && go test -race ./...

# Manual test (if applicable)

```

## Checklist

- [ ] C++ changes: `clang-format` applied (`find cpp -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i`)
- [ ] Go changes: `gofmt` and `go vet` clean
- [ ] New features: test cases added
- [ ] Performance changes: benchmark numbers included in description
- [ ] Breaking API change: noted in description and migration path provided
