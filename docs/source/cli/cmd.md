# Commands

The following examples are typical uses of the commands.
Each command has its own options, to see all the options please check [this section](#help-message).

:::{tip}
By default the output name is deduced from the parameters and it is placed in the current directory, you can use the option `-o` to change it.
:::

## Infer

The `infer` sub command infers a network parameter from a dataset.
The command can be run by typing:

```console
$ harissa infer path/to/dataset
```

:::{note}
For backward compatibility the [old format](https://github.com/ulysseherbach/harissa?tab=readme-ov-file#basic-usage) is supported.

```console
$ harissa infer path/to/dataset.txt
```

:::

To choose a method of inference, you need to add **after the options of `infer`**
the name of the wanted method (followed by more options).

```console
$ harissa infer path/to/dataset hartree
$ harissa infer path/to/dataset hartree --verbose
```

## Trajectory

The `trajectory` command simulate a trajectory of a single cell.
It needs a path to the simulation parameters and a network parameter.
You can run the command by typing:

```console  
$ harissa trajectory path/to/simulation_param path/to/network_param
```

To choose a method of simulation, you need to add **after the options of `trajectory`**
the name of the wanted method (followed by more options).

```console
$ harissa trajectory path/to/simulation_param path/to/network_param ode
$ harissa trajectory path/to/simulation_param path/to/network_param ode --verbose
```

## Dataset

The `dataset` command simulate `**TODO** description`.
The output dataset will have the same cell number and the same time points than input dataset.

It needs a path to a dataset and a network parameter.
You can run the command by typing:

```console 
$ harissa dataset path/to/dataset path/to/network_param
```

To choose a method of simulation, you need to add **after the options of `dataset`**
the name of the wanted method (followed by more options).

```console
$ harissa dataset path/to/dataset path/to/network_param ode
$ harissa dataset path/to/dataset path/to/network_param ode --verbose
```

## Convert

This `convert` command convert an `npz` file to folder containing `txt` arrays and vice versa.

It takes 
You can run the command by typing:

```console 
$ harissa convert path/to/file
```

If you want to decide the output path you should run:

```console
$ harissa convert path/to/file output/path
```

:::{note}
Normally it accepts only `npz` file and directory but it also allows you to convert an [old format](https://github.com/ulysseherbach/harissa?tab=readme-ov-file#basic-usage) dataset (an `txt` file) to the new format (`npz` or folder containing `txt`).

```console
$ harissa convert path/to/dataset.txt
```
:::

## Visualize

The `visualize` command compare 2 datasets and export the comparison.

It takes a reference dataset and a dataset.
You can run the command by typing:

```console
$ harissa visualize path/to/ref_dataset path/to/dataset
```

## Template

The `template` command is a helper to create your custom `Inference` or `Simulation` subclass.
The command generates a template python file.
You can generate 2 types of template:

- inference
- simulation

The template contains a subclass of either `Inference` or `Simulation` and is 
ready to be used, however the subclass is a dummy with minimal implementation.
Feel free to modify the implementation of `__init__`, `run` and `directed` 
functions to fit your need.

You can run the command by typing:

```console
$ harissa template simulation path/to/my_simulation
```

or

```console
$ harissa template inference path/to/my_inference
```

## Arguments from file

With all these options, the command line can be long and it become exhausting to modify it inside the terminal.

To ease it, you can write the arguments inside a `.txt` file.

Then you can run:

```console
$ harissa @args.txt
```

See [here](help.ipynb) to have more informations about commands options.
