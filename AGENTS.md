When I say look at the project, you start with docs/. Progress documentation is regularly added
which help understand the project state. This is supposed to be a monorepo for the project and
its webpage/site.

The Hugo project page lives in site/. To build: hugo --source site --destination public.
The Python project uses uv with hatchling build backend. Scripts live in scripts/.
