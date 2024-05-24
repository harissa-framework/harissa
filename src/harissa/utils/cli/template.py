from pathlib import Path
from re import sub

template_codes = {
    'simulation' : lambda class_name: f'''
"""
Simulation template
"""

import numpy as np
from harissa.core.simulation import Simulation

class {class_name}(Simulation):
    """
    {class_name} simulation.
    """
    def __init__(self):
        """
        This is a dummy init. Feel free to change its implementation or 
        to delete it.
        """ 
        self.verbose = False

    def run(self, time_points, initial_state, parameter):
        """
        Perform a constant simulation. Feel free to change its implementation.

        Parameters
        ----------
        time_points: np.ndarray
            Recorded time points.

        initial_state: np.ndarray
            initial rna and protein levels.

        parameter: NetworkParameter
            network parameter used to simulate (here not used).

        Returns
        -------
        Simulation.Result
        """
        states = np.repeat(initial_state[np.newaxis,...], time_points.size, 0)
        return self.Result(time_points, states[:, 0], states[:, 1])

'''.lstrip(),
    'inference': lambda class_name: f'''
"""
Inference template
"""

import numpy as np
from harissa.core.parameter import NetworkParameter
from harissa.core.inference import Inference
from harissa.core.dataset import Dataset

class {class_name}(Inference):
    """
    {class_name} inference.
    """
    def __init__(self):
        """
        This is a dummy init. Feel free to change its implementation or 
        to delete it.
        """ 
        self.verbose = False

    @property
    def directed(self):
        """
        True if the {class_name} infers an 
        asymmetric interaction matrix else False.

        Returns
        -------
        bool
        """
        return True

    def run(self, data, parameter):
        """
        Perform an identity inference. Feel free to change its implementation.

        Parameters
        ----------
        data: Dataset
            dataset used to infer (here not used)

        parameter: NetworkParameter
            network parameter updated after inferring

        Returns
        -------
        Inference.Result
            Wrapper class around the updated network and other custom data.

        """
        return self.Result(parameter)

'''.lstrip()
}

def template(args):
    if args.path.absolute() == Path.cwd().absolute():
        path = args.path / args.path.absolute().name
    else:
        path = args.path

    path = path.with_suffix('.py') 
    code = template_codes[args.template_type](
        sub(r'[_-]+', ' ', path.stem).title().replace(' ', '')
    )

    with open(path, 'w') as python_file:
        python_file.write(code)

def add_subcommand(main_subparsers):
    parser = main_subparsers.add_parser(
        'template',
        help='generate a template for Inference or Simulation subclass'
    )
    parser.add_argument(
        'template_type',
        choices=tuple(template_codes.keys()),
        help='Type of template that will be generated.'
    )
    parser.add_argument(
        'path', 
        type=Path,
        help='path where to generate the template. '
             'The name of the template is deducted from it.'
    )

    parser.set_defaults(run=template)
