# Command line interface

:::{toctree}
---
maxdepth: 2
---
cmd
help
:::

The harissa package provides a command line interface (cli).
Once the package is installed, a `harissa` program can be run from your terminal.
It allows people to use the package main functionalities without having to code any python lines.
Of course, you are welcome to use the [python API](../api/index.md) if the cli doesn't satisfy your needs.

The cli provides 6 commands:

* infer
* trajectory
* dataset
* visualize
* convert
* template

See [here](cmd.md) to get the full documentation about the commands.

## Format file

The package `harissa` use [numpy arrays](https://numpy.org/doc/stable/reference/arrays.html) to infer or simulate.

The following commands need to access arrays through files. 
`Numpy` provides file formats to export or import arrays:

* `npz`: multiple arrays inside 1 binary (compressed) file
* `npy`: 1 array inside a binary file
* `txt`: 1 array inside a text file

We decided to accept only  `npz` file or folder containing `txt` files.
We kept the `txt` files format because it allows to edit arrays easily.

:::{tip}
By default the output format is `.npz`, this format is compacted but it is not editable.
To generate editable output you can use the option `-f txt` for all the commands except convert.

You can also use the convert command to do it.

Once you finish the modification to some data you can convert it back to `.npz` thanks to the convert command.
:::
