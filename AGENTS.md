When the user asks a question that involves this repository,
start your exploration with the docs/ folder. The documentation
there should be enough in most cases as opposed to screening
the entire repository.

Anything python related should be handled using uv + pyproject.toml

Consider uv add instead of uv pip install when dealing with new packages.

When creating a doc, follow the YYYY-MM-DD_01_n-a-m-e.md syntax.

The Hugo project page lives in site/. To build: hugo --source site --destination public.
The Python project uses uv with hatchling, src layout (src/probenet/). Scripts live in scripts/.

Whenever you implement a plan, also create a doc with all the details for future reference.