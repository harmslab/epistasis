# Documentation Guide

Documentation was created using Sphinx. They are hosted on ReadTheDocs.com.

## Building the Docs

Use the following steps to build the docs.

1. Clone the documentation locally
2. Add your documentation
3. If adding a new file, make sure to list it in the table-of-contents of the
`index.rst` file.
4. Build the docs locally by calling `make html`.
5. Commit changes and pull request to master repo.

## Example Gallery

The example gallery is created using []*sphinx-gallery*](https://github.com/sphinx-gallery/sphinx-gallery).
This will attempt to run any `plot_` prefixed python files in the `../examples/py`
directory and turn them into a demo page. See the sphinx-gallery documentation
pages to format such examples.   

If you are creating an example, please build the docs locally on your machine and commit
the changes to our master repo. ReadTheDocs will not build these on their server.

## API Documentation

The API documentation was created using the `autodoc` Sphinx extension. This pulls
docstrings directly from the source code.

If you are adding documentation to the API, add it to the source code directly. Next,
make sure the module/package/class is referenced properly in the appropriate
autodoc files inside the `api` folder. If you are adding a new module to the source
code, add a new API autodoc reference following the syntax in the `api` folder (or see
the [`sphinx.ext.autodoc`](http://www.sphinx-doc.org/en/stable/ext/autodoc.html) documentation).
