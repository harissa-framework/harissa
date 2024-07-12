# Building the documentation

Use python `3.10` or `3.11` but not `3.12` because `sphinxcontrib.collections`
depends on something that is deprecated and removed in `3.12`.

## Install dependencies

From the root of the project you can install the dependencies required for the 
documentation generation by typing:

```bash
$ pip install -r docs/requirements.txt
```

## Build

### Single version

To build the single version website run the command:

```bash
$ sphinx-build -M html docs/source docs/build
```

or go the `docs` folder and run `make html`.

### Multi version

To build the multi version website run the command:

```bash
$ sphinx-multiversion docs/source harissa
```

## Visualise the website

### Single version

You can open the `docs/build/index.html` in your browser or start a local server[^1]
and go to the build folder.

### Multi version

The output dir must be `harissa` for the version dropdown to work.

0. If needed rename your output folder to `harissa`.

1. Start a local server[^1] at the root


2. Go to the folder harissa


[^1]: To start a local server you can use python `http.server` or [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) if you are using vscode.
    
    ```bash
    python3 -m http.server
    ```
