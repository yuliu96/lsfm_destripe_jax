import requirements

with open("requirement_jax.txt", "r") as fd:
    for req in requirements.parse(fd):
        print(req)
